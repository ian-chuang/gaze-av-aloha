#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn, Tensor
import einops
import numpy as np


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()
    return return_observations