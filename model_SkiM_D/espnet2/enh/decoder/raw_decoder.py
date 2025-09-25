import math

import torch

from .abs_decoder import AbsDecoder


class RawDecoder(AbsDecoder):
    """Transposed Convolutional decoder for speech enhancement and separation"""

    def __init__(
        self,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        # assert kernel_size // 2 == stride

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        """
        input = input.transpose(1, 2)
        batch_size = input.shape[0]
        padding = 0

        output_size = (self.kernel_size - ilens.max())%self.stride + ilens.max()

        # B, D, T
        wav = torch.nn.functional.fold(input, output_size=(output_size, 1), kernel_size=(self.kernel_size, 1), dilation=1, stride=(self.stride, 1), padding=(padding, 0))
        div = torch.nn.functional.fold(torch.ones_like(input), output_size=(output_size, 1), kernel_size=(self.kernel_size, 1), dilation=1, stride=(self.stride, 1), padding=(padding, 0))
        wav = wav / (div + 1e-6)
        wav =wav.squeeze(1).squeeze(2) 

        wav = wav[:, 0:ilens.max()]


        return wav, ilens




if __name__ == "__main__":
    # from espnet2.enh.encoder.raw_encoder import RawEncoder
    from raw_encoder import RawEncoder

    input_audio = torch.randn((2, 9957))
    ilens = torch.LongTensor([9957, 98])

    kernel_size = 32
    stride = 16

    encoder = RawEncoder(kernel_size=kernel_size, stride=stride,)
    decoder = RawDecoder(kernel_size=kernel_size, stride=stride,)
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)


    torch.testing.assert_close(wav, input_audio)
