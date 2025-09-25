import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class NullEncoder(AbsEncoder):
    """Null encoder."""

    def __init__(self, expand_dim=False):
        super().__init__()
        self.expand_dim = expand_dim

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        """
        if self.expand_dim:
            input = input.unsqueeze(2)

        return input, ilens
