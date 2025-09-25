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
from espnet2.enh.layers.SkiMDiff import SkiMDiff
from espnet2.enh.layers.dcunet import DCUNet
from espnet2.enh.layers.ncsnpp import NCSNpp
from espnet2.train.class_choices import ClassChoices
from espnet2.enh.layers.skim import SkiMComplex
from espnet2.enh.loss.criterions.time_domain import SDRLoss

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if not isinstance(out, tuple):
            out = [out]
        
        for o in out:

            nan_mask = torch.isnan(o)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", o[nan_mask.nonzero()[:, 0].unique(sorted=True)])



score_choices = ClassChoices(
    name="score_model",
    classes=dict(dcunet=DCUNet, ncsnpp=NCSNpp, skimc=SkiMComplex, skim=SkiMDiff),
    type_check=torch.nn.Module,
    default=None,
)

sde_choices = ClassChoices(
    name="sde",
    classes=dict(
        ouve=OUVESDE,
        ouvp=OUVPSDE,
    ),
    type_check=SDE,
    default="ouve",
)


class ScoreModel(AbsDiffusion):
    def __init__(self, normalize_input=False, discriminative=False, use_history=False, zero_his=None, oracle_his = False, num_eval_batch=10, **kwargs):
        super().__init__()

        score_model = kwargs["score_model"]
        score_model_class = score_choices.get_class(kwargs["score_model"])
        self.dnn = score_model_class(**kwargs["score_model_conf"])
        self.sde = sde_choices.get_class(kwargs["sde"])(**kwargs["sde_conf"])
        self.loss_type = "mse" if "loss_type" not in kwargs else kwargs['loss_type']
        self.t_eps = 3e-2 if "t_eps" not in kwargs else kwargs['t_eps']
        self.normalize_input = normalize_input
        self.use_history = use_history
        self.discriminative = discriminative
        self.online_size = getattr(self.dnn, 'online_size', -1)

        self.num_eval_batch = num_eval_batch
        self.evaled = 0
        # for submodule in self.modules():
        #     submodule.register_forward_hook(nan_hook)
        self.sisdr = SDRLoss()
        self.zero_his = zero_his
        self.oracle_his = oracle_his
        if self.zero_his:
                print(f'use his {self.zero_his}')


    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "mae":
            losses = err.abs()
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "logmse":
            losses = torch.log(torch.square(err.abs()))
            loss = torch.mean(0.5 * torch.mean(losses.reshape(losses.shape[0], -1), dim=-1))

        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        return loss

    def get_pc_sampler(
        self, predictor_name, corrector_name, y,  hc=None, N=None, states=None, **kwargs
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_pc_sampler(
            predictor_name,
            corrector_name,
            sde=sde,
            score_fn=self.score_fn,
            y=y,
            hc=hc,
            states=states,
            **kwargs
        )



    def get_ode_sampler(self, y, N=None, states=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_ode_sampler(
            sde, self.score_fn, y=y, device=y.device, states=states, **kwargs
        )


    def score_fn(self, x, t, y, hx=None, states=None):
        # Concatenate y as an extra channel

        if hx is not None:
            if self.zero_his:
                if self.zero_his == 'rand':
                    hx = torch.rand_like(hx)
                elif self.zero_his == 'id':
                    hx = x
                else:
                    hx = torch.zeros_like(hx)
            dnn_input = torch.cat([x, y, hx], dim=1)
        else:
            dnn_input = torch.cat([x, y,], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo

        if states is not None:
            # streaming mode
            out, states = self.dnn.forward_stream(dnn_input, t, states=states)
            score = - out
        else:
            score = -self.dnn(dnn_input, t)
        
        
        return score

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F

        # B, C, F, T
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)

        if self.normalize_input:
            denominator = y.abs().max() * 1.1 + 1e-5
            x = x / denominator
            y = y / denominator

        if self.use_history:
            if self.discriminative:
                h_x = x
            else:
                h_x = torch.nn.functional.pad(x, (self.online_size, 0))
                h_x = h_x[..., 0:-self.online_size]
            if self.oracle_his:
                h_x = x
        else:
            h_x = None
 

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self.score_fn(perturbed_data, t, y, hx=h_x)
        assert score.shape == z.shape
        err = score * sigmas + z
        loss = self._loss(err)

        stats = {}

        if self.discriminative:
            des_output = self.dnn(y, t=t, des=True)
            des_si_sdr = self.sisdr(x.squeeze(), des_output.squeeze()).mean()

            loss += des_si_sdr
            stats = {'si_sdr_des': - des_si_sdr.detach()}

        if self.training:
            self.evaled = 0
        else:
            self.evaled += 1
        if not self.training and self.evaled < self.num_eval_batch:
            x = feature_ref
            y = feature_mix
            if self.use_history:
                h_x = torch.nn.functional.pad(feature_ref, (0, 0, self.online_size, 0))
                h_x = h_x[:, 0:-self.online_size, :]
                if self.oracle_his:
                    h_x = x
            else:
                h_x = None
            enhanced = self.enhance(y, h_x=h_x)
            si_sdr = - self.sisdr(x.squeeze(), enhanced.squeeze()).mean().detach()
            stats['si_sdr'] =  si_sdr


        return loss, stats

    def enhance_streaming(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        h_x = None,
        states = None,
        timeit=False,
        **kwargs      
    ):


        Y = noisy_specturm.unsqueeze(1)
        if h_x is not None:
            h_x = h_x.unsqueeze(1)

        if sampler_type == "pc":

            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                hc=h_x,
                states=states,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, states=states, N=N, **kwargs)

        X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1)


        return X_Hat

    def enhance_des(self, noise):
        if self.normalize_input:
            denominator = noise.abs().max() * 1.1 + 1e-5
            noise = noise / denominator


        t = (
            torch.rand(noise.shape[0], device=noise.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )

        noise = noise.unsqueeze(1).permute(0, 1, 3, 2)
        des_output = self.dnn(noise, t=t, des=True)
        des_output = des_output.squeeze(1).permute (0, 2, 1)
        des_output = des_output / (des_output.abs().max()) * 0.9

        return des_output


    def enhance(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        h_x = None,
        timeit=False,
        **kwargs
    ):
        if self.online_size != -1:
            noisy_specturm = torch.nn.functional.pad(noisy_specturm, (0, 0, self.online_size * 4, 0), mode='reflect')


        Y = noisy_specturm.permute(0, 2, 1).unsqueeze(1)
        if h_x is not None:
            h_x = torch.nn.functional.pad(h_x, (0, 0, self.online_size * 4, 0), mode='reflect')
            h_x = h_x.permute(0, 2, 1).unsqueeze(1)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                hc=h_x,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))

        X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)

        if self.online_size != -1:
            X_Hat = X_Hat[:, self.online_size * 4:, :]

        return X_Hat
