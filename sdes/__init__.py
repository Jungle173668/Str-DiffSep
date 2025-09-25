# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
import functools
import math

import torch
import torch.nn as nn
from scipy import integrate

from .correctors import Corrector, CorrectorRegistry
from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor


import tqdm

__all__ = [
    "PredictorRegistry",
    "CorrectorRegistry",
    "Predictor",
    "Corrector",
    "get_sampler",
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


@functools.cache
def fibonaccispace(start, end, steps, device=None):
    fib_num = [0, 1]
    while len(fib_num) < steps:
        fib_num.append(fib_num[-1] + fib_num[-2])

    fib_max = fib_num[-1]
    fib_num = [fib / fib_max for fib in fib_num]
    t = torch.tensor(fib_num, device=device).cumsum()
    t = t / t[-1]
    t = t * (end - start) + start
    return t
    # return torch.tensor(fib_num, device=device)


def get_pc_scheduled_sampler(
    predictor_name,
    corrector_name,
    sde,
    score_fn,
    y,
    denoise=True,
    true_mean=None,
    eps=3e-2,
    snr=0.1,
    corrector_steps=1,
    probability_flow: bool = False,
    intermediate=False,
    schedule="linear",
    **kwargs,
):
    """Create a Predictor-Corrector (PC) sampler with scheduled step size
    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.
    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler():
        """The PC sampler function."""
        if intermediate:
            im = []
        with torch.no_grad():
            if true_mean is not None:
                xt = sde.prior_sampling(true_mean.shape, true_mean).to(true_mean.device)
            else:
                xt = sde.prior_sampling(y.shape, y).to(y.device)
            # timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            base = 10
            if schedule == "linear":
                timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)
            elif schedule == "log":
                timesteps = torch.logspace(
                    math.log(sde.T) / math.log(base),
                    math.log(eps) / math.log(base),
                    sde.N + 1,
                    base=base,
                    device=y.device,
                )
            elif schedule == "revlog":
                timesteps = torch.logspace(
                    math.log(eps) / math.log(base),
                    math.log(sde.T) / math.log(base),
                    sde.N + 1,
                    base=base,
                    device=y.device,
                ).flip(dims=(0,))
            else:
                raise NotImplementedError(f"Schedule '{schedule}' does not exist")
            # timesteps = 1 - timesteps
            # timesteps = timesteps.flip(dims=(0,)) + eps
            for i in range(sde.N):
                t = timesteps[i]
                dt = abs(timesteps[i] - timesteps[i + 1])
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, vec_t, y, dt=dt)
                if intermediate:
                    im.append((xt, xt_mean))
                xt, xt_mean = predictor.update_fn(xt, vec_t, y, dt=dt)
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            if intermediate:
                return x_result, ns, im
            else:
                return x_result, ns

    return pc_sampler


def get_pc_sampler(
    predictor_name,
    corrector_name,
    sde,
    score_fn,
    y,
    true_mean=None,
    denoise=True,
    eps=3e-2,
    snr=0.1,
    corrector_steps=1,
    probability_flow: bool = False,
    intermediate=False,
    skim_inference=None,
    **kwargs,
):
    """Create a Predictor-Corrector (PC) sampler.
    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.
    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    
    def pc_sampler():
        """The PC sampler function."""
        if intermediate:
            im = []
        with torch.no_grad():
            if true_mean is not None:
                xt = sde.prior_sampling(true_mean.shape, true_mean).to(true_mean.device)
            else:
                xt = sde.prior_sampling(y.shape, y).to(y.device)

            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)

            for i in range(sde.N):
                # print("this is step ", i)
                t = timesteps[i]
                vec_t = torch.ones(y.shape[0], device=y.device) * t

                xt, xt_mean = corrector.update_fn(xt, vec_t, y, )
                if intermediate:
                    im.append((xt, xt_mean))
                xt, xt_mean = predictor.update_fn(xt, vec_t, y, )


            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            if intermediate:
                return x_result, ns, im
            else:
                return x_result, ns

    return pc_sampler


def get_ode_sampler(
    sde,
    score_fn,
    y,
    inverse_scaler=None,
    denoise=True,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=3e-2,
    device="cuda",
    **kwargs,
):
    """Probability flow ODE sampler with the black-box ODE solver.
    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.
    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, vec_eps, y)
        return x

    def drift_fn(x, t, y):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, t, y)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.
        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
                **kwargs,
            )
            nfe = solution.nfev
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(y.shape)
                .to(device)
                .type(torch.complex64)
            )

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler



def get_pc_sampler_multidiff(
    predictor_name,
    corrector_name,
    sde,
    score_fn,
    backbone,
    y,
    true_score_fn=None,
    true_mean=None,
    denoise=True,
    eps=3e-2,
    snr=0.1,
    corrector_steps=1,
    probability_flow: bool = False,
    intermediate=False,
    **kwargs,
):
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)
    backbone.seg_overlap = True

    original_online_size = backbone.online_size
    online_size = original_online_size

    
    def pc_sampler():
        """The streaming PC sampler function."""
        # Cut chunks
        # print(snr, "snr in streaming sampler")
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)  # [B, 2, T]
            # print(xt.shape)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)

            # start diffusion steps
            # for n in range(sde.N):
            for n in tqdm.tqdm(range(sde.N), desc="Diffusion steps"):
                # t vector
                t = timesteps[n]
                vec_t = torch.ones(xt.shape[0], device=y.device) * t

                # Prepare xt for backbone
                xt = torch.cat((xt, y), dim=1)  # [B, 2+1, T]
                xt = xt.unsqueeze(2)  # [B, 2+1, 1, T]

                # Use backbone to cut the input into chunks
                
                # print(backbone.segment_size)
                # print(backbone.online_size)
                backbone.online_size = online_size
                xt_frames, rest = backbone.frame_raw(xt)  # [B, 3, K, S]

                # print(xt_frames.shape)
                backbone.online_size = original_online_size

                B, _, _, S = xt_frames.shape
                # print(xt_frames.shape)

                outputs = []
                states = {}

                # for s in range(S):
                    # states = {}
                for s in tqdm.tqdm(range(S), desc="Segments", leave=False):
                    # use list to collect outputs for each segment

                    # Extract input frame
                    input_frame = xt_frames[:, :, :, s]
                    xt = input_frame[:, :2, :]  # [B, 2, 1, T]
                    mix = input_frame[:, 2:3, :]  # [B, 1, 1, T]
                    # print(xt.shape, mix.shape)

                    # corrector step
                    states['p_or_c'] = 'c'
                    xt, xt_mean = corrector.update_fn(xt, vec_t, mix, states)
                    # print(states.keys())
                    # print(states['p_or_c'])

                    # predictor step
                    states['p_or_c'] = 'p'
                    xt, xt_mean = predictor.update_fn(xt, vec_t, mix, states)
                    # print(states.keys())
                    # print(states['p_or_c'])
                    
                    # get the result for this segment and diffusion step (s, n)
                    x_result = xt_mean if denoise else xt  # [B, 2, T]
                    outputs.append(x_result)  # [S, B, 2, T]

                    # print(states.keys())
                    
                
                output_channels = outputs[0].shape[1] # C=2
                ori_len = y.shape[-1]  # T


                outputs_tensor = torch.stack(outputs, dim=0)  # [S, B, C, T]
                outputs_tensor = outputs_tensor.permute(1, 2, 0, 3)  # [B, C, S, T]

                # merge segments by channel
                all_channels_results = []

                backbone.online_size = online_size
                for ch in range(output_channels):
                    # [B, S, T]
                    ch_segments = outputs_tensor[:, ch, :, :]
                    # merge_raw
                    ch_merged = backbone.merge_raw(ch_segments, ori_len)  # [B, T]
                    all_channels_results.append(ch_merged)  #[[B,T], [B,T]]
                backbone.online_size = original_online_size

                xt = torch.stack(all_channels_results, dim=1)  # [B, C, T]

            ns = sde.N * (corrector.n_steps + 1)

            return xt, ns
        
    return pc_sampler




def get_pc_sampler_multidiff_equal(
    predictor_name,
    corrector_name,
    sde,
    score_fn_equal,
    score_fn_full,
    score_stream,
    backbone,
    y,
    true_mean=None,
    denoise=True,
    eps=3e-2,
    snr=0.1,
    corrector_steps=1,
    probability_flow: bool = False,
    skim_inference=None,
    intermediate=False,
    **kwargs,
):
    
    sde.N = 30
    snr = 0.5
    corrector_steps = 2

    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn_equal, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn_equal, snr=snr, n_steps=corrector_steps)

    # # Test no states version (Uncomment to use, and change update_fn's 'score_bs' to 'states')
    # predictor = predictor_cls(sde, score_stream, probability_flow=probability_flow)
    # corrector = corrector_cls(sde, score_stream, snr=snr, n_steps=corrector_steps)


    # test consistency between full and stream score
    def frame_raw(input, kernel):
        B, C, _, ori_len = input.shape
        
        stride = kernel // 2
        to_pad = (kernel - ori_len)%stride

        padded = torch.nn.functional.pad(input, (0, to_pad, 0, 0), mode='reflect')
        padded = padded.transpose(2, 3) #B, C, T, 1

        frame = torch.nn.functional.unfold(
            padded, kernel_size=(kernel, 1), stride=(stride, 1),
        )

        frame = frame.view(B, C, kernel, -1)

        return frame, to_pad

    def merge_raw(output, ori_len, kernel):
        stride = kernel // 2
        to_pad = (kernel - ori_len)%stride

        output_size = to_pad + ori_len
        output = output.permute(0, 2, 1)

        wav = torch.nn.functional.fold(output, output_size=(output_size, 1), kernel_size=(kernel, 1), dilation=1, stride=(stride, 1), padding=(0, 0))
        div = torch.nn.functional.fold(torch.ones_like(output), output_size=(output_size, 1), kernel_size=(kernel, 1), dilation=1, stride=(stride, 1), padding=(0, 0))
        wav = wav / (div + 1e-6)
        wav =wav.squeeze(1).squeeze(2)

        wav = wav[:, :ori_len]

        return wav


    # [B, S, C, K] -> [B*S, C, K]
    def _merge_bs(x, C, B, S, K):
        # x: [B, C, K, S] -> [B, S, C, K] -> [B*S, C, K]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * S, C, K)
        return x

    def merge_by_channel(x_bs, B, S, C, K, ori_len):
        x_temp = x_bs.view(B, S, C, K).permute(0, 2, 3, 1).contiguous()  # [B, C, K, S]
        x_temp_channels = []
        for ch in range(C):
            ch_segments = x_temp[:, ch, :, :]
            ch_segments_for_merge = ch_segments.permute(0, 2, 1)  # [B, S, K]
            backbone.seg_overlap = True
            ch_merged = backbone.merge_raw(ch_segments_for_merge, ori_len)
            # ch_merged = merge_raw(ch_segments_for_merge, ori_len, 200)
            x_temp_channels.append(ch_merged)
        
        x_t_temp = torch.stack(x_temp_channels, dim=1)  # [B, C, T]
        return x_t_temp
        

    def pc_sampler():
        backbone.seg_overlap = True

        """The streaming PC sampler function."""
        use_chunk = True
        test_consistency = True
        # print(backbone)
    
        # Cut chunks
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)  # [B, 2, T]

            # print(xt.shape)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)

            # test consistency between full and stream score
            if test_consistency:
                model = backbone
                model.eval()
                
                input = torch.randn(4, 3,  1, 8000).cuda()  # [B, C, 1, T]
                t = torch.rand(4, device=input.device)

                # complete output
                seg_output = model(input, t)

                # framed output
                frames, rest = model.frame_raw(input)
                B, _, _, S = frames.shape  # B, C, K, S

                state = {}
                outputs = []

                state['p_or_c'] = 'p'
                for i in range(S):
                    input_frame = frames[:, :, :, i]  # (B, C, K)
                    input_frame = input_frame.unsqueeze(2)  # (B, C, 1, K)

                    output, state = model.forward_stream(input_frame,t, states=state)  # (B, C, 1, K)
                
                    outputs.append(output.squeeze(2))  # (B, C, K)
                

                output_channels = output.shape[1]
                ori_len = input.shape[-1]

                all_channel_results = []

                for ch in range(output_channels):
                    ch_segments = []
                    for seg_output_item in outputs:
                        ch_segments.append(seg_output_item[:, ch, :])  # (B, TT) List.

                    ch_all_segments = torch.stack(ch_segments, dim=1)  # (B, S, TT)  Stack.

                    # merge on single channel
                    ch_merged = model.merge_raw(ch_all_segments, ori_len)
                    all_channel_results.append(ch_merged)

                if output_channels == 1:
                    stream_final = all_channel_results[0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, ori_len)
                else:
                    stream_final = torch.stack(all_channel_results, dim=1).unsqueeze(2)  # (B, output_channels, 1, ori_len)
                
                # print(seg_output.shape, stream_final.shape)

                torch.testing.assert_close(seg_output, stream_final, atol=0.015, rtol=0.015)
                print("streaming ok")



            # start diffusion steps
            for n in range(sde.N):
            # for n in tqdm.tqdm(range(sde.N), desc="Diffusion steps"):
                # t vector
                t = timesteps[n]
                vec_t = torch.ones(xt.shape[0], device=y.device) * t

                # calculate score on full sequence
                full_score = score_fn_full(xt, vec_t, y)  # [B, 2, T]                

                # Prepare xt for backbone
                if use_chunk == True:
                    xt = torch.cat((xt, y), dim=1)  # [B, 2+1, T]
                    xt = xt.unsqueeze(2)  # [B, 2+1, 1, T]


                    # Use backbone to cut the input into chunks
                    backbone.seg_overlap = True
                    xt_frames, rest = backbone.frame_raw(xt)  # [B, 3, K, S]
                    # xt_frames, rest = frame_raw(xt, 200)  # [B, 3, K, S]

                    backbone.seg_overlap = True
                    full_score_frames, rest = backbone.frame_raw(full_score.unsqueeze(2))  # [B, 2, K, S]
                    # full_score_frames, rest = frame_raw(full_score.unsqueeze(2), 200)  # [B, 2, K, S]
                    # B, _, _, S = xt_frames.shape

                    inputs_2ch = xt_frames[:, :2, :, :]            # [B, 2, K, S]
                    mix_1ch     = xt_frames[:, 2:3, :, :]           # [B, 1, K, S]
                    score_all   = full_score_frames                  # [B, 2, K, S]
                    

                    B, _, K, S = inputs_2ch.shape
                    ori_len = y.shape[-1]
                    output_channels = 2

                    xt_bs    = _merge_bs(inputs_2ch, 2, B, S, K)   # [B*S, 2, K]
                    mix_bs   = _merge_bs(mix_1ch, 1, B, S, K)    # [B*S, 1, K]
                    score_bs = _merge_bs(score_all, 2, B, S, K)   # [B*S, 2, K]

                    # vec_t: [B] -> [B, 1] -> repeat S -> [B, S] -> flatten -> [B*S]
                    vec_t_bs = vec_t.view(B, 1).expand(B, S).reshape(B * S)

                    # corrector step
                    # xt_bs, xt_mean_bs = corrector.update_fn(xt_bs, vec_t_bs, mix_bs, score_bs)  # [B*S, 2, K]                     
                    # x_t_temp = merge_by_channel(xt_bs, B, S, output_channels, K, ori_len)  # [B, 2, T]
                    
                    # full_score = score_fn_full(x_t_temp, vec_t, y)
                    # full_score_frames, rest = backbone.frame_raw(full_score.unsqueeze(2))  # [B, 2, K, S]
                    # score_bs = _merge_bs(full_score_frames, 2, B, S, K)   # [B*S, 2, K]
                    states = {}

                    for step in range(1):
                        states = {}
                        states['p_or_c'] = 'c'
                        xt_bs, xt_mean_bs = corrector.update_fn(xt_bs, vec_t_bs, mix_bs, score_bs)  # [B*S, 2, K]                     
                        
                        # reconstruct score
                        x_t_temp = merge_by_channel(xt_bs, B, S, output_channels, K, ori_len)  # [B, 2, T]
                        
                        full_score = score_fn_full(x_t_temp, vec_t, y)

                        backbone.seg_overlap = True
                        full_score_frames, rest = backbone.frame_raw(full_score.unsqueeze(2))  # [B, 2, K, S]
                        # full_score_frames, rest = frame_raw(full_score.unsqueeze(2), 200)  # [B, 2, K, S]
                        score_bs = _merge_bs(full_score_frames, 2, B, S, K)   # [B*S, 2, K]

                    # predictor step
                    states = {}
                    states['p_or_c'] = 'p'
                    xt_bs, xt_mean_bs = predictor.update_fn(xt_bs, vec_t_bs, mix_bs, score_bs)

                    # select output
                    x_result_bs = xt_mean_bs if denoise else xt_bs   # [B*S, 2, K]

                    xt = merge_by_channel(x_result_bs, B, S, output_channels, K, ori_len)
                    xt_mean = xt

                else:
                    full_score = score_fn_full(xt, vec_t, y)
                    xt, xt_mean = corrector.update_fn(xt, vec_t, y, score_bs)
                    # xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                    # if intermediate:
                        # im.append((xt, xt_mean))
                    full_score = score_fn_full(xt, vec_t, y)
                    xt, xt_mean = predictor.update_fn(xt, vec_t, y, score_bs)
                    # xt, xt_mean = predictor.update_fn(xt, vec_t, y)

            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)

            return x_result, ns
        
    return pc_sampler


