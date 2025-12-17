#!/usr/bin/env python3
# geomlang_viz_basic.py
#
# Minimal, "back to basics" visualization helpers for geomlang.
# Assumes input images have 3 channels:
#   ch0: red fill
#   ch1: blue fill
#   ch2: edges

import torch

def channels_to_rgb(x):
    """
    x: tensor [B, 3, H, W] or [3, H, W] in [0, 1]
    Returns: rgb [B, 3, H, W] in [0, 1]
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = x.clamp(0.0, 1.0)

    red  = x[:, 0]          # [B,H,W]
    blue = x[:, 1]
    edge = x[:, 2]

    # pure red/blue; edges as light gray
    R = red + 0.7 * edge
    G = 0.7 * edge
    B = blue + 0.7 * edge

    rgb = torch.stack([R, G, B], dim=1)
    rgb = rgb.clamp(0.0, 1.0)
    return rgb
