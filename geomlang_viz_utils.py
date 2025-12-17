#!/usr/bin/env python3
# Shared viz utilities.

import numpy as np
import torch


def channels_to_crisp_rgb(img_3chw, bg_thresh=0.15):
    """
    Convert decoder output [3, H, W] in [0,1] (red, blue, edge) to uint8 RGB.
    Background = black; red/blue fills to pure colors; edges to white.
    """
    if isinstance(img_3chw, np.ndarray):
        img = torch.from_numpy(img_3chw)
    else:
        img = img_3chw.detach().cpu()
    img = img.clamp(0.0, 1.0)

    red = img[0]
    blue = img[1]
    edge = img[2]

    H, W = red.shape

    bg_mask = (red < bg_thresh) & (blue < bg_thresh) & (edge < bg_thresh)

    stacked = torch.stack([red, blue, edge], dim=0)  # [3, H, W]
    winner = stacked.argmax(dim=0)                   # [H, W] in {0,1,2}

    rgb = torch.zeros(3, H, W, dtype=torch.float32)
    rgb[0][winner == 0] = 1.0            # red
    rgb[2][winner == 1] = 1.0            # blue
    rgb[:, winner == 2] = 1.0            # edge -> white

    for c in range(3):
        rgb[c][bg_mask] = 0.0

    return (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
