from typing import Callable
from functools import wraps
from thop import profile, clever_format


import time
import warnings
import logging
import torch
import transformers
import os
import json
import random


def calculate_inference_time(verbosity: bool=False):
    """
    Decorator which calculate the inference time in seconds.

    Args:
        verbosity (bool): If False, ignore warnings and useless logging messages.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not verbosity:
                warnings.filterwarnings("ignore")
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            start_time = time.time()
            outputs = func(*args, **kwargs)
            torch.cuda.synchronize()  # Wait for all GPU tasks to complete
            end_time = time.time()
            return end_time - start_time, outputs
        return wrapper
    return decorator


def calculate_flops(
    model,
    batch_size: int=1,
    sys_token_len: int=11,
    seq_len: int=1024,
    image_size: int=336,
):
    if type(model) == transformers.LlavaForConditionalGeneration:
        vision_token_len = (image_size // model.config.vision_config.patch_size) ** 2
        pixel_values = torch.randn(
            batch_size,
            3,
            image_size,
            image_size,
            device=model.device,
        )
    elif type(model) == transformers.Qwen2_5_VLForConditionalGeneration:
        vision_token_len = (image_size // model.config.vision_config.patch_size // model.config.vision_config.spatial_merge_size) ** 2
        t_grid = 1
        h_grid = int(image_size / model.config.vision_config.patch_size)
        w_grid = int(image_size / model.config.vision_config.patch_size)
        pixel_values = torch.randn(
            h_grid * w_grid * batch_size,
            model.config.vision_config.patch_size ** 2 * 2 * 3,
            device=model.device,
        )
        image_grid_thw = torch.tensor(
            [[t_grid, h_grid, w_grid]],
            dtype=torch.long,
            device=model.device,
        )

    input_ids = torch.cat((
        torch.arange(sys_token_len, device=model.device),
        torch.full((vision_token_len,), model.config.image_token_id, device=model.device),
        torch.arange(seq_len - sys_token_len - vision_token_len, device=model.device)
    )).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.full_like(input_ids, 1, dtype=torch.long, device=model.device)
    position_ids = torch.arange(
        seq_len,
        dtype=torch.long,
        device=model.device
    ).expand(batch_size, -1)
    cache_position = position_ids.view(1, -1).squeeze(0)

    if type(model) == transformers.LlavaForConditionalGeneration:
        dummy_inputs = (
            input_ids,
            pixel_values,
            attention_mask,
            position_ids,
        )
    elif type(model) == transformers.Qwen2_5_VLForConditionalGeneration:
        dummy_inputs = (
            input_ids,
            attention_mask,
            position_ids.expand(4, -1, -1),
            None,
            None,
            None,
            True,
            False,
            False,
            pixel_values,
            None,
            image_grid_thw,
            None,
            None,
            cache_position,
        )

    flops, params = profile(model, inputs=dummy_inputs, verbose=False)
    flops, params = clever_format([flops, params], "%.2f")

    return flops, params