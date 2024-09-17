import torch
from PIL import Image
from pathlib import Path
from typing import Any, Tuple
from enum import Enum
import numpy as np
import einops

class Modality(Enum):
    SAR = 'radar'
    OPTICAL_LOW_RES = 'lores'
    OPTICAL_HIGH_RES = 'hires'


def load_model_checkpoint(checkpoint: Path, model: Any):
    assert checkpoint.exists()
    
    state_dict = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(state_dict)
    
    return model


def resize_array_raw(image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    # load to image, resize, return to np array
    image = Image.fromarray(image_array)
    resized_image = image.resize(target_size)
    resized_array = np.array(resized_image)
    
    return resized_array


def resize_array(
        image_array: np.ndarray,
        target_size: Tuple[int, int],
        channel_first: bool = False
    ) -> np.ndarray:
    
    # if needed, rearrance to HWC order
    if channel_first:
        image_array = einops.rearrange(image_array, 'c h w -> h w c')
    
    
    channels = image_array.shape[-1]
    
    if channels == 3:
        # 3 channels is supported natively
        image_array = resize_array_raw(image_array, target_size)
    else:
        # do this channel-by-channel
        rescaled_channels = []
        for ch in range(channels):
            rescaled_channels.append(
                resize_array_raw(image_array[:,:,ch], target_size)
            )
                    
        image_array, _ = einops.pack(rescaled_channels, 'h w *')
    
    
    # undo HWC if we applied it
    if channel_first:
        image_array = einops.rearrange(image_array, 'h w c -> c h w')
        
        
    return image_array


def add_blank_channel(
        image_array: np.ndarray,
        channel_pos: int = 0,
        channel_fill: int = 0,
        channel_first: bool = False
    ):
    
    # if needed, rearrance to HWC order
    if channel_first:
        image_array = einops.rearrange(image_array, 'c h w -> h w c')
    
    
    channels = image_array.shape[-1]
    
    dummy_channel = np.ones_like(image_array[:,:,0]) * channel_fill
    # do this channel-by-channel
    all_channels = []
    for ch in range(channels):
        all_channels.append(
            image_array[:,:,ch]
        )
        

    
    # hacky but apparently fast
    all_channels.insert(channel_pos, dummy_channel) 
    # all_channels[channel_pos:channel_pos] = dummy_channel
    
    image_array, _ = einops.pack(all_channels, 'h w *')
    
    # undo HWC if we applied it
    if channel_first:
        image_array = einops.rearrange(image_array, 'h w c -> c h w')
        
        
    return image_array


def scale_channels(
        image_array: np.ndarray,
        global_min: float | None = None,
        global_max: float | None = None,
        new_max: float = 255.0
    ):
    
    if global_min is None:
        global_min = np.min(image_array)
    if global_max is None:
        global_max = np.max(image_array)
    
    assert global_min <= global_max

    if (global_max - global_min) == 0:
        return np.zeros_like(image_array) # skip a step

    # force cast these to floats
    global_min = float(global_min)
    global_max = float(global_max)
    
    # we'll be working with floats
    original_dtype = image_array.dtype
    image_array = image_array.astype(float)
    
    # to the scaling!
    image_array -= global_min
    image_array[image_array<0.0] = 0.0
    image_array /= (global_max - global_min)
    image_array[image_array>1.0] = 1.0
    image_array *= new_max
    
    return image_array.astype(original_dtype)
