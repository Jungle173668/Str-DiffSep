import math

import torch

from .abs_encoder import AbsEncoder


class RawEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        # assert kernel_size // 2 == stride



        self.stride = stride
        self.kernel_size = kernel_size

        # self.unfold = torch.nn.Unfold(kernel_size=(kernel_size,1), dilation=1, padding=(self.stride, 0), stride=(stride, 1))
        self.unfold = torch.nn.Unfold(kernel_size=(kernel_size,1), dilation=1, padding=(0, 0), stride=(stride, 1))
        self._output_dim = kernel_size

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = input[:, None, :, None]

        max_len = input.shape[2]
        to_pad = (self.kernel_size - max_len)%self.stride
        input = torch.nn.functional.pad(input, (0, 0, 0, to_pad,), mode='reflect')

        feature = self.unfold(input)
        feature = feature.transpose(1, 2)

        to_pad = (self.kernel_size - ilens)%self.stride
        flens = (
            torch.div(ilens + to_pad - self.kernel_size, self.stride, rounding_mode="floor") + 1
        )
        return feature, flens




if __name__ == "__main__":
    input_audio = torch.randn((2, 100))
    ilens = torch.LongTensor([100, 98])

    nfft = 32
    win_length = 28
    hop = 10

    encoder = RawEncoder(kernel_size=nfft, stride=hop)
    frames, flens = encoder(input_audio, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]

    sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_allclose(sframes, frames)
