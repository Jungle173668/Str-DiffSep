from collections import OrderedDict
from typing import Dict, Tuple
from espnet2.enh.decoder.raw_decoder import RawDecoder
from espnet2.enh.encoder.raw_encoder import RawEncoder
from espnet2.enh.layers.skim import MemLSTM, SegLSTM, SkiM
from espnet2.enh.separator.abs_separator import AbsSeparator


import torch
import torch.nn as nn


class SkiMDiff_Separator(AbsSeparator):
    def __init__(self, output_dim=None, **kwargs) -> None:
        super().__init__()

        self.skim = SkiMDiff(des=True, **kwargs)

    def forward(self, input, ilens, additional):
        # B, T, 1 
        # to 
        # B, c, 1, T
        B, T, _ = input.shape
        input = input.view(B, T, 1 , 1).permute(0, 2, 3, 1)
        out = self.skim(input, None)

        out = out.squeeze(1).permute(0, 2, 1)

        return [out, ], ilens, {}
    @property
    def num_spk(self):
        return 1

class SkiMDiff(SkiM):


    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
        kernel_size=16,
        stride=8,
        with_conformer=False,
        c_ffn=1024,
        c_att_head=4,
        c_kernel=7,
        use_history=False,
        des = False,
    ):

        super().__init__(
            hidden_size, hidden_size, output_size, dropout, num_blocks, segment_size,bidirectional,
            mem_type, norm_type, seg_overlap,
        )

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=True,
                    norm_type=norm_type,
                    with_conformer=with_conformer,
                    c_ffn=c_ffn,
                    c_att_head=c_att_head,
                    c_kernel=c_kernel,
                )
            )
 
        self.des = des
        self.input_fc = nn.Linear(input_size, hidden_size)
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size if bidirectional else hidden_size*2,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                        use_t_emb=False if self.des else True,
                    )
                )
        self.encoder = RawEncoder(kernel_size=kernel_size, stride=stride)
        self.decoder = RawDecoder(kernel_size=kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.online_size = (self.segment_size - 1) * stride + kernel_size
        self.use_history = use_history

    def forward(self, input, t):
        # input shape: B, T (S*K), D

        # B, c, 1, T
        # 4, 3, 1, T
        ori_len = input.shape[-1]
        frames, rest = self.frame_raw(input)
        '''
        B: batch size
        C: channel number
        TT: segment size
        S: segment number
        '''
        B, C, TT, S = frames.shape  # B, Channel, segment size, segment num 
        K = self.segment_size
        D = self.kernel_size
        frames = frames.permute(0, 1, 3, 2).contiguous()  # B, C, S, TT 
        frames = frames.view(B*C*S, TT)
        ilens = (torch.ones(frames.shape[0])*TT).long()
        frames, _ = self.encoder(frames, ilens=ilens) # B*C*S, K, D   # length is TT, and is reshaped to K,D
        frames = frames.view(B, C, S, K, D).permute(0, 2, 3, 1, 4).contiguous().view(B, S, K, C*D)


        input = self.input_fc(frames)

        # B, S, K, D 

        D = self.hidden_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                if self.des:
                    hc = self.mem_lstms[i](hc, S)
                else:
                    hc = self.mem_lstms[i](hc, S, t)
                pass
        # output = output.view(B, S, K, D)


        output = self.output_fc(output.permute(0, 2, 1)).permute(0, 2, 1) # B*S, K, D

        ilens = (torch.ones(output.shape[0])*TT).long()

        output, _ = self.decoder(output, ilens)
        output = output.view(B, S, TT)


        output = self.merge_raw(output, ori_len).view(B, 1, 1, ori_len)

        return output


    def empty_seg_states(self, device=None, dtype='float32', batch_size=-1):
        shp = (2, batch_size, self.hidden_size)
        return (
            torch.zeros(*shp, device=device, dtype=dtype),
            torch.zeros(*shp, device=device, dtype=dtype),
        )
    def forward_stream(self, input, t, states):
        # signal slightly differ in the tail, due to different padding methods of 
        # `forward` and `forward_stream`

        xx = input[:, 0, 0, :]
        yy = input[:, 1, 0, :]
        if self.use_history:
            hx = input[:, 2, 0, :]

        ori_len = xx.shape[1]
        ilens= (torch.ones(xx.shape[0])*ori_len).long()
        xx = self.encoder(xx, ilens)[0]
        yy = self.encoder(yy, ilens)[0]
        if self.use_history:
            hx = self.encoder(hx, ilens)[0]
            input = torch.cat([xx, yy, hx], dim=2)
        else:
            input = torch.cat([xx, yy], dim=2)

        # B, C, L, T = input.shape

        # input = input.view(B, C*L, T).permute(0, 2, 1)
 
        input = self.input_fc(input)

        B, K, D = input.shape

        state_key = f"stage_{states['p_or_c']}_t_{t.mean().item():.3f}"
        if state_key not in states:
            states[state_key] = self.init_state(input.device, input.dtype, B)

        output = input

        for i in range(self.num_blocks):
            output, states[state_key]["seg_state"][i] = self.seg_lstms[i](
                output, states[state_key]["seg_state"][i]
            )

        tmp_states = [self.empty_seg_states(input.device, input.dtype, B) for i in range(self.num_blocks)]

        for i in range(self.num_blocks - 1):
            tmp_states[i + 1], states[state_key]["mem_state"][i] = self.mem_lstms[
                i
            ].forward_one_step(states[state_key]["seg_state"][i], states[state_key]["mem_state"][i], t)

        states[state_key]["seg_state"] = tmp_states
        states[state_key]['current_step'] += 1

        output = self.output_fc(output.permute(0, 2, 1)).transpose(1, 2)

        output = self.decoder(output, ilens=ilens.long())[0].unsqueeze(1).unsqueeze(1)
        return output, states

    def init_state(self, device, dtype, batch_size):
        states = {
                "current_step": 0,
                "seg_state": [self.empty_seg_states(device, dtype, batch_size) for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        return states

    def frame_raw(self, input):
        # B, C, 1, T
        B, C, _, ori_len = input.shape


        kernel = self.online_size
        stride = self.online_size // 2 if self.seg_overlap else self.online_size
        to_pad = (kernel - ori_len)%stride

        padded = torch.nn.functional.pad(input, (0, to_pad, 0, 0), mode='reflect')
        padded = padded.transpose(2, 3) #B, C, T, 1

        frame = torch.nn.functional.unfold(
            padded, kernel_size=(kernel, 1), stride=(stride, 1),
        )

        frame = frame.view(B, C, kernel, -1)


        return frame, to_pad

    def merge_raw(self, output, ori_len):
        # B, T, D
        kernel = self.online_size
        stride = self.online_size // 2 if self.seg_overlap else self.online_size
        to_pad = (kernel - ori_len)%stride

        output_size = to_pad + ori_len

        output = output.permute(0, 2, 1)

        wav = torch.nn.functional.fold(output, output_size=(output_size, 1), kernel_size=(kernel, 1), dilation=1, stride=(stride, 1), padding=(0, 0))
        div = torch.nn.functional.fold(torch.ones_like(output), output_size=(output_size, 1), kernel_size=(kernel, 1), dilation=1, stride=(stride, 1), padding=(0, 0))
        wav = wav / (div + 1e-6)
        wav =wav.squeeze(1).squeeze(2)

        wav = wav[:, :ori_len]

        return wav


if __name__ == "__main__":
    torch.manual_seed(111)

    

    model = SkiMDiff(
        48,
        11,
        16,
        dropout=0.0,
        num_blocks=4,
        segment_size=20,
        bidirectional=False,
        kernel_size =16,
        stride=8,
        mem_type="hc",
        norm_type="cLN",
        seg_overlap=False,
        use_history=True,
    )
    model.eval()
    seq_len = 9527

    input = torch.randn(4, 3,  1, seq_len)
    t = torch.rand(4, device=input.device)
    seg_output = model(input, t)


    frames, rest = model.frame_raw(input)

    B, _, _, S = frames.shape
    # frames,ilens, T, rest = model.frame(input)
    # B, S, K, D = frames.shape
    state = {}
    outputs = []
    for i in range(S):
        input_frame = frames[:, :, :, i]
        input_frame = input_frame.unsqueeze(2)
        output, state = model.forward_stream(input_frame,t, states=state)
        outputs.append(output)
    output = torch.cat(outputs, dim =2).squeeze(1)


    stream_out = model.merge_raw(output,seq_len)
    torch.testing.assert_allclose(seg_output, stream_out[:, None, None, :])
    print("streaming ok")


