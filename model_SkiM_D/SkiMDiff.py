from collections import OrderedDict
from typing import Dict, Tuple
from .espnet2.enh.decoder.raw_decoder import RawDecoder
from .espnet2.enh.encoder.raw_encoder import RawEncoder
from .espnet2.enh.layers.skim_copy import MemLSTM, SegLSTM, SkiM
from .espnet2.enh.separator.abs_separator import AbsSeparator


import torch
import torch.nn as nn
import time


class SkiMDiff(SkiM):

    def __init__(
        self,
        num_sources,
        hidden_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="cLN",
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

        input_size = (num_sources + 1) * kernel_size
        output_size = num_sources * kernel_size

        super().__init__(
            input_size, hidden_size, output_size, dropout, num_blocks, segment_size,bidirectional,
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

        self.input_size = input_size
        self.output_size = output_size

        self.input_fc = nn.Linear(self.input_size, hidden_size)

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

        self.output_fc = nn.Sequential(nn.PReLU(), nn.Conv1d(hidden_size, self.output_size, 1))


            

    def forward(self, input, t):
        # B, c, 1, T
        # 4, 3, 1, T
        input_channels = input.shape[1]
        output_channels = input_channels - 1
 
        ori_len = input.shape[-1]
        frames, rest = self.frame_raw(input)
        '''
        B: batch size
        C: channel number
        TT: segment size
        S: segment number
        '''
        B, C, TT, S = frames.shape  # B, Channel, segment size(online size), segment num 
        K = self.segment_size
        D = self.kernel_size
        frames = frames.permute(0, 1, 3, 2).contiguous()  # B, C, S, TT 
        frames = frames.view(B*C*S, TT)
        ilens = (torch.ones(frames.shape[0])*TT).long()
        frames, _ = self.encoder(frames, ilens=ilens) # B*C*S, K, D   # length is TT, and is reshaped to K,D
        frames = frames.view(B, C, S, K, D).permute(0, 2, 3, 1, 4).contiguous().view(B, S, K, C*D)

        input = self.input_fc(frames)  # B, S, K, Hidden_size = D
        
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
                
        # output = output.view(B, S, K, D)
        output = self.output_fc(output.permute(0, 2, 1)).permute(0, 2, 1)  # BS, K, C*D
        multi_ch_output = output.view(B*S, K, self.kernel_size, output_channels)

        all_outputs = []
        for ch in range(output_channels):
            ch_output = multi_ch_output[:, :, :, ch].contiguous()
            ch_ilens = torch.full((ch_output.shape[0],), TT, dtype=torch.long)
            ch_decoded, _ = self.decoder(ch_output, ilens=ch_ilens)
            ch_decoded = ch_decoded.view(B, S, TT)
            ch_merged = self.merge_raw(ch_decoded, ori_len)

            all_outputs.append(ch_merged)

        output = torch.stack(all_outputs, dim=1)
        output = output.unsqueeze(2)  # (B, output_channels, 1, ori_len)

        return output


    def empty_seg_states(self, device=None, dtype='float32', batch_size=-1):
        shp = (2, batch_size, self.hidden_size)
        return (
            torch.zeros(*shp, device=device, dtype=dtype),
            torch.zeros(*shp, device=device, dtype=dtype),
        )
    

    def init_state(self, device, dtype, batch_size):
        states = {
                "current_step": 0,
                "seg_state": [self.empty_seg_states(device, dtype, batch_size) for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        return states

    def forward_stream(self, input, t, states):
        # signal slightly differ in the tail, due to different padding methods of 
        # `forward` and `forward_stream`

        input_channels = input.shape[1]
        output_channels = input_channels - 1

        xx = input[:, 0, 0, :]
        yy = input[:, 1, 0, :]
        mixx = input[:, 2, 0, :]
        if self.use_history:
            hx = input[:, 3, 0, :]
            # print("set hx to ones")
            # hx = torch.randn_like(hx)

        ori_len = xx.shape[1]
        ilens= (torch.ones(xx.shape[0])*ori_len).long()

        xx = self.encoder(xx, ilens)[0]
        yy = self.encoder(yy, ilens)[0]
        mixx = self.encoder(mixx, ilens)[0]

        # if self.use_history:
        #     hx = self.encoder(hx, ilens)[0]
        #     output_his = self.input_fc_his(hx)

        input = torch.cat([xx, yy, mixx], dim=2)

        # B, C, L, T = input.shape

        # input = input.view(B, C*L, T).permute(0, 2, 1)

        input = self.input_fc(input)

        B, K, D = input.shape

        state_key = str(t.cpu())
        state_key = f"stage_{states['p_or_c']}_t_{t.mean().item():.3f}"
        
        if state_key not in states:
            states[state_key] = self.init_state(input.device, input.dtype, B)

        output = input

        for i in range(self.num_blocks):

            output, _hc = self.seg_lstms[i](
                output, states[state_key]["seg_state"][i]
            )
            if self.use_history and i < self.num_blocks -1:
                output_his, hc_his = self.seg_lstms_his[i](output_his, _hc)
                _hc = (_hc[0]+ hc_his[0], _hc[1]+hc_his[1])
            states[state_key]["seg_state"][i] = _hc

        tmp_states = [self.empty_seg_states(input.device, input.dtype, B) for i in range(self.num_blocks)]

        for i in range(self.num_blocks - 1):
            tmp_states[i + 1], states[state_key]["mem_state"][i] = self.mem_lstms[i].forward_one_step(
                states[state_key]["seg_state"][i], states[state_key]["mem_state"][i], t)

        states[state_key]["seg_state"] = tmp_states
        states[state_key]['current_step'] += 1
        # print("Current step:", states[state_key]['current_step'])

        output = self.output_fc(output.permute(0, 2, 1)).transpose(1, 2)  # B, K, C*D
        
        # multi-output
        multi_ch_output = output.view(B, K, self.kernel_size, output_channels)

        all_outputs = []
        for ch in range(output_channels):
            ch_output = multi_ch_output[:, :, :, ch].contiguous()
            ch_decoded = self.decoder(ch_output, ilens=ilens.long())[0]  # (B, ori_len)
            all_outputs.append(ch_decoded)

        if output_channels == 1:
            final_output = all_outputs[0].unsqueeze(1).unsqueeze(1)
        else:
            final_output = torch.stack(all_outputs, dim=1).unsqueeze(2)

        # output = self.decoder(output, ilens=ilens.long())[0].unsqueeze(1).unsqueeze(1)

        return final_output, states


    def frame_raw(self, input):
        # B, C, 1, T
        # print(input.shape)
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
        # B, S, TT
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
        num_sources=2,  # num of speakers
        hidden_size=256,
        dropout=0.0,
        num_blocks=8,
        segment_size=50,
        bidirectional=False,
        kernel_size=16,
        stride=8,
        mem_type="hc",
        norm_type="cLN",
        seg_overlap=True,
        use_history=False,
    )

    model.eval()

    model = model

    print(model)
    
    input = torch.randn(4, 3,  1, 8000)
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
    
    print(seg_output.shape, stream_final.shape)

    torch.testing.assert_close(seg_output, stream_final)
    print("streaming ok")
