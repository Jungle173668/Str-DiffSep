# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT


import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

import espnet2.enh.diffusion.sampling as sampling
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.sdes import OUVESDE, OUVPSDE, SDE
from espnet2.train.class_choices import ClassChoices
from espnet2.enh.diffusion.score_based_diffusion import ScoreModel
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class HistoryNet(torch.nn.Module):

    def __init__(self, streaming_size):
        super().__init__()
        self.streaming_size = streaming_size

        self.rnn = RNN(
            idim=512,
            elayers=3,
            cdim=384,
            hdim=256,
            dropout=0.3,
            typ='lstm'
        )

    def forward(self, x, priv_state=None):
        # input x: B, T, F
        B, T, F = x.shape
        x = torch.cat([x.real, x.imag], dim=2).permute(0, 2, 1)

        x = torch.nn.functional.pad(x, (self.streaming_size, 0))[:, :, :-self.streaming_size,].permute(0, 2, 1)

        out, _, state = self.rnn(x, [T]*B, priv_state)


        return out, state



class StreamingScoreModel(ScoreModel):
    def __init__(self, streaming_size, use_history=False, **kwargs):

        kwargs['score_model_conf']['history'] = use_history
        super().__init__(**kwargs)

        self.streaming_size = streaming_size

        self.use_history = use_history
        if use_history:
            self.history = HistoryNet(streaming_size)

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def score_fn(self, x, t, y, hc=None):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t, hc)
        return score

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)


        B, _, F, T = x.shape
        CT = self.streaming_size
        assert T % CT == 0
        N = T // CT

        x_c = x.view(B, 1, F, N, CT).permute(0, 3, 1,2, 4).view(B*N, 1, F, CT)
        y_c = y.view(B, 1, F, N, CT).permute(0, 3, 1,2, 4).view(B*N, 1, F, CT)
        x, y = x_c, y_c

        if self.use_history:
            history, _ = self.history(feature_ref)
            history = history.permute(0, 2, 1).unsqueeze(1)
            h_c = history.view(B, 1, F, N, CT).permute(0, 3, 1,2, 4).view(B*N, 1, F, CT)
        else:
            h_c = None

        

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self.score_fn(perturbed_data, t, y, h_c)
        err = score * sigmas + z
        loss = self._loss(err)

        return loss

    def pad_input(self, x):
        # B, T, F
        B, ori_T, F = x.shape

        x = x.permute(0, 2, 1)

        to_pad = self.streaming_size - ori_T % self.streaming_size

        x = torch.nn.functional.pad(x, (0, to_pad)).permute(0, 2, 1)
        return x, ori_T


    def enhance(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        timeit=False,
        **kwargs
    ):
        Y, ori_T = self.pad_input(noisy_specturm)

        

        Y = Y.permute(0, 2, 1).unsqueeze(1)
        B, _, F, T = Y.shape
        CT = self.streaming_size
        assert T % CT == 0
        NN = T // CT

        if self.use_history:
            Y = Y.view(B, 1, F, NN, CT).permute(0, 3, 1,2, 4).unbind(dim=1)
            c_x = torch.zeros(B, CT, F) + 1j * torch.zeros(B, CT, F)
            c_x = c_x.to(noisy_specturm.device)
            state = None
            X_hats = []
            for i, yy in enumerate(Y):
                hc, state = self.history(c_x, state)
                hc = hc.permute(0, 2, 1).unsqueeze(1)

                sampler = self.get_pc_sampler(predictor, corrector, yy, hc=hc, N=N if i > 2 else 30, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
                x_out, nfe = sampler()
                c_x = x_out.squeeze(1).permute(0, 2, 1)
                X_hats.append(x_out)
                
            X_Hat = torch.cat(X_hats, dim=3)
        else:
            Y = Y.view(B, 1, F, NN, CT).permute(0, 3, 1,2, 4).view(B*NN, 1, F, CT)      

            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )

            X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)
        X_Hat = X_Hat.view(B, NN*CT, F)[:, 0:ori_T, :]

        return X_Hat

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, hc=None, N=None, minibatch=None, **kwargs
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self.score_fn,
                y=y,
                hc=hc,
                **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self.score_fn,
                        y=y_mini,
                        hc=hc,
                        **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn