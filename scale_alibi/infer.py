import torch
from pathlib import Path

from scale_alibi.util import load_model_checkpoint
from scale_alibi.model import ScaleAlibi
from scale_alibi.croma.pretrain_croma import CROMA
from scale_alibi.train import ScaleAlibiParams, CromaParams
from scale_alibi.dataset.loader import MultimodalSample, LoresMultimodalSample

# --- MODEL LOADING ---


def load_croma(model_path: Path, params: CromaParams | None = None):
    if params is None:
        params = CromaParams(
            radar_dataset_path = None,
            lores_dataset_path = None
            
            # all defaults
        )

    croma = CROMA(
        # ... defaults should be okay
        # except for total_channels, we may need to remove channels from S2,
        total_channels=params.total_channels, # 2 sar, 3 lores_visual
        num_patches=params.num_patches,
        patch_size=params.patch_size
    )
    
    croma = load_model_checkpoint(model_path, croma)
    
    return croma


def load_scale_alibi(model_path: Path, params: ScaleAlibiParams | None = None):
    if params is None:
        params = ScaleAlibiParams(
            radar_dataset_path=None,
            lores_dataset_path=None,
            hires_dataset_path=None
            
            # all defaults
        )
    
    salibi = ScaleAlibi(
        total_channels=params.total_channels,
        num_patches=params.num_patches,
        patch_size=params.patch_size
    )
    
    salibi = load_model_checkpoint(model_path, salibi)
    
    return salibi
