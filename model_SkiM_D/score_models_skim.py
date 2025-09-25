import torch
import torchaudio
from hydra.utils import instantiate

import sys
import os
import site
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pl_model import DiffSepModel


from espnet2.bin.enh_inference_streaming import SeparateSpeechStreaming
from espnet2.bin.enh_inference import SeparateSpeech

import soundfile as sf
from IPython.display import Audio
import torch
import tqdm
import time

import numpy as np

from omegaconf import OmegaConf


class ScoreModelSkiM(torch.nn.Module):
    def __init__(
        self,
        backbone_args):
        
        super().__init__()
        

        self.backbone = instantiate(backbone_args)

    

    def forward(self, xt, time_cond, mix, skim_inference=None):
        """
        Args:
            xt: (batch, channels, time)
            time_cond: (batch,)
            mix: (batch, channels, time)
        Returns:
            x: (batch, channels, time) same size as input
        """
        x = torch.cat((xt, mix), dim=1)  # (batch, num_sources + 1, time)
        x = x.unsqueeze(2)  # (batch, num_scources + 1, 1, time)

        self.backbone.seg_overlap = True
        x = self.backbone(x, time_cond)  # (batch, num_sources, 1, time)
        
        x = x.squeeze(2)  # (batch, num_sources, time)
        
        return x    
    
    def forward_streaming(self, xt, time_cond, mix, states, skim_inference=None):

        x = torch.cat((xt, mix), dim=1)  # (batch, num_sources + 1, time)
        x = x.unsqueeze(2)  # (batch, num_sources + 1, 1, time)

        self.backbone.seg_overlap = True
        x, states = self.backbone.forward_stream(x, time_cond, states)  # (batch, num_sources, 1, time)

        x = x.squeeze(2)  # (batch, num_sources, time)

        return x

    def forward_equal(self, xt, time_cond, mix, equal_score):

        return equal_score



if __name__ == "__main__":
    pass
