#!/usr/bin/env python3
# geomlang_latent_edit.py
#
# Multishape + relation + scale + shape model
# PLUS a "latent edit playground" that nudges the concept manifold:
#   - left_of -> right_of
#   - red_smaller -> red_larger
#   - red_circle -> red_square
#
# Outputs:
#   - training logs in console
#   - PNGs in outputs_latent_edit/

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------
# Hyperparameters
# ------------------------
IMG_SIZE     = 32
N_CHANNELS   = 2          # red, blue
LATENT_DIM   = 48
BATCH_SIZE   = 256
EPOCHS       = 300        # you can bump this up if you want
LR           = 1e-3
N_SAMPLES    = 12000      # dataset size

RELATIONS    = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
SCALES       = ["red_larger", "red_smaller", "similar"]
SHAPES       = ["circle", "square"]

N_REL   = len(RELATIONS)
N_SCALE = len(SCALES)
N_SHAPE = len(SHAPES)   # for red and blue separately

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------
# Simple drawing helpers
# ------------------------

def draw_circle(img, cx, cy, r, channel):
    yy, zz = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = (yy - cy) ** 2 + (zz - cx) ** 2 <= r ** 2
    img[channel][mask] = 1.0


def draw_square(img, cx, cy, half_side, channel):
    y0 = max(0, cy - half_side)
    y1 = min(IMG_SIZE, cy + half_side)
    x0 = max(0, cx - half_side)
    x1 = min(IMG_SIZE, cx + half_side)
    img[channel, y0:y1, x0:x1] = 1.0

def generate_scene(rel_idx, scale_idx, red_shape_idx, blue_shape_idx):
    """
    Generate a single 2-channel image with:
      - a red object in channel 0
      - a blue object in channel 1
    Relational constraints:
      left_of/right_of/above/below/inside/overlapping
    Scale constraints:
      red_larger / red_smaller / similar
    Shapes: circle or square for each color.

    This version is careful to NEVER give np.random.randint an invalid range.
    """
    rel   = RELATIONS[rel_idx]
    scale = SCALES[scale_idx]
    red_shape  = SHAPES[red_shape_idx]
    blue_shape = SHAPES[blue_shape_idx]

    img = np.zeros((2, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    margin = 4

    # --- choose radii safely ---
    base_r = np.random.randint(4, 7)  # base size 4..6

    if scale == "red_larger":
        r_red  = base_r + np.random.randint(2, 4)   # bigger
        r_blue = base_r + np.random.randint(-1, 1)  # around base
    elif scale == "red_smaller":
        r_red  = base_r + np.random.randint(-1, 1)
        r_blue = base_r + np.random.randint(2, 4)
    else:  # similar
        r_red  = base_r + np.random.randint(-1, 2)
        r_blue = base_r + np.random.randint(-1, 2)

    # clamp radii
    r_red  = int(np.clip(r_red, 3, 10))
    r_blue = int(np.clip(r_blue, 3, 10))

    # helper: safe randint with fallback
    def safe_randint(low, high):
        """
        Return np.random.randint(low, high_inclusive+1) if possible.
        If high <= low, collapse to a single valid point (low) and return that.
        """
        low = int(low)
        high = int(high)
        if high <= low:
            return low
        return np.random.randint(low, high + 1)

    # helper: sample a center for a given radius within full image
    def sample_center(radius):
        cx = safe_randint(margin + radius, IMG_SIZE - margin - radius)
        cy = safe_randint(margin + radius, IMG_SIZE - margin - radius)
        return cx, cy

    # declare centers
    cx_red = cy_red = cx_blue = cy_blue = None

    # ----------------------------
    # Relation-specific placement
    # ----------------------------
    if rel in ["left_of", "right_of"]:
        # Split the image vertically into left and right halves,
        # but keep bounds safe with radius+margin.
        # Left side:
        left_x_min  = margin + r_red
        left_x_max  = min(IMG_SIZE // 2 - margin, IMG_SIZE - margin - r_red)
        if left_x_max <= left_x_min:
            # fallback: use global range
            left_x_min = margin + r_red
            left_x_max = IMG_SIZE - margin - r_red

        # Right side:
        right_x_min = max(IMG_SIZE // 2 + margin, margin + r_blue)
        right_x_max = IMG_SIZE - margin - r_blue
        if right_x_max <= right_x_min:
            right_x_min = margin + r_blue
            right_x_max = IMG_SIZE - margin - r_blue

        # y is free for both, just keep them valid
        y_red_min   = margin + r_red
        y_red_max   = IMG_SIZE - margin - r_red
        y_blue_min  = margin + r_blue
        y_blue_max  = IMG_SIZE - margin - r_blue

        if rel == "left_of":
            cx_red  = safe_randint(left_x_min, left_x_max)
            cy_red  = safe_randint(y_red_min, y_red_max)
            cx_blue = safe_randint(right_x_min, right_x_max)
            cy_blue = safe_randint(y_blue_min, y_blue_max)
        else:  # right_of
            cx_blue = safe_randint(left_x_min, left_x_max)
            cy_blue = safe_randint(y_blue_min, y_blue_max)
            cx_red  = safe_randint(right_x_min, right_x_max)
            cy_red  = safe_randint(y_red_min, y_red_max)

    elif rel in ["above", "below"]:
        # Split the image horizontally into top and bottom halves.
        # Common x region for both:
        common_x_min = margin + max(r_red, r_blue)
        common_x_max = IMG_SIZE - margin - max(r_red, r_blue)
        if common_x_max <= common_x_min:
            common_x_min = margin + max(r_red, r_blue)
            common_x_max = IMG_SIZE - margin - max(r_red, r_blue)

        # top (for "above")
        top_y_min  = margin + r_red
        top_y_max  = min(IMG_SIZE // 2 - margin, IMG_SIZE - margin - r_red)
        if top_y_max <= top_y_min:
            top_y_min = margin + r_red
            top_y_max = IMG_SIZE - margin - r_red

        # bottom (for "below")
        bottom_y_min = max(IMG_SIZE // 2 + margin, margin + r_blue)
        bottom_y_max = IMG_SIZE - margin - r_blue
        if bottom_y_max <= bottom_y_min:
            bottom_y_min = margin + r_blue
            bottom_y_max = IMG_SIZE - margin - r_blue

        if rel == "above":
            cx_red  = safe_randint(common_x_min, common_x_max)
            cy_red  = safe_randint(top_y_min, top_y_max)
            cx_blue = safe_randint(common_x_min, common_x_max)
            cy_blue = safe_randint(bottom_y_min, bottom_y_max)
        else:  # below
            cx_blue = safe_randint(common_x_min, common_x_max)
            cy_blue = safe_randint(top_y_min, top_y_max)
            cx_red  = safe_randint(common_x_min, common_x_max)
            cy_red  = safe_randint(bottom_y_min, bottom_y_max)

    elif rel == "inside":
        # We define "inside" as one object strictly containing the other.
        # Decide which is outer, then place inner at same center.
        if scale == "red_smaller":
            # blue outer, red inner (if necessary, fix radii)
            if r_blue <= r_red:
                r_blue = min(10, r_red + 2)
            # sample a center for blue
            cx_blue, cy_blue = sample_center(r_blue)
            cx_red, cy_red   = cx_blue, cy_blue
        else:
            # red outer (for red_larger or similar)
            if r_red <= r_blue:
                r_red = min(10, r_blue + 2)
            cx_red, cy_red   = sample_center(r_red)
            cx_blue, cy_blue = cx_red, cy_red

    else:  # overlapping
        # Start from red, then place blue nearby with a small offset.
        cx_red, cy_red = sample_center(r_red)

        # small jitter, at most half the smaller radius
        r_min = min(r_red, r_blue)
        dx = np.random.randint(-r_min // 2, r_min // 2 + 1)
        dy = np.random.randint(-r_min // 2, r_min // 2 + 1)

        cx_blue = int(np.clip(cx_red + dx, margin + r_blue, IMG_SIZE - margin - r_blue))
        cy_blue = int(np.clip(cy_red + dy, margin + r_blue, IMG_SIZE - margin - r_blue))

    # ----------------------------
    # Draw shapes
    # ----------------------------
    if red_shape == "circle":
        draw_circle(img, cx_red, cy_red, r_red, 0)
    else:
        draw_square(img, cx_red, cy_red, r_red, 0)

    if blue_shape == "circle":
        draw_circle(img, cx_blue, cy_blue, r_blue, 1)
    else:
        draw_square(img, cx_blue, cy_blue, r_blue, 1)

    return img


def generate_dataset(n_samples=N_SAMPLES):
    scenes      = np.zeros((n_samples, N_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    rel_labels  = np.zeros((n_samples,), dtype=np.int64)
    scale_labels= np.zeros((n_samples,), dtype=np.int64)
    red_shapes  = np.zeros((n_samples,), dtype=np.int64)
    blue_shapes = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        rel_idx   = np.random.randint(0, N_REL)
        scale_idx = np.random.randint(0, N_SCALE)
        red_idx   = np.random.randint(0, N_SHAPE)
        blue_idx  = np.random.randint(0, N_SHAPE)

        img = generate_scene(rel_idx, scale_idx, red_idx, blue_idx)

        scenes[i]      = img
        rel_labels[i]  = rel_idx
        scale_labels[i]= scale_idx
        red_shapes[i]  = red_idx
        blue_shapes[i] = blue_idx

    return (
        torch.from_numpy(scenes),
        torch.from_numpy(rel_labels),
        torch.from_numpy(scale_labels),
        torch.from_numpy(red_shapes),
        torch.from_numpy(blue_shapes),
    )


# ------------------------
# Model
# ------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 4, 2, 1),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),          # 64x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),         # 128x4x4
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, N_CHANNELS, 4, 2, 1),  # 2x32x32
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        x_recon = self.deconv(h)
        return x_recon


class GeomMulti(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        self.rel_head   = nn.Linear(latent_dim, N_REL)
        self.scale_head = nn.Linear(latent_dim, N_SCALE)
        self.shapeR_head= nn.Linear(latent_dim, N_SHAPE)
        self.shapeB_head= nn.Linear(latent_dim, N_SHAPE)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)

        rel_logits   = self.rel_head(z)
        scale_logits = self.scale_head(z)
        shapeR_logits= self.shapeR_head(z)
        shapeB_logits= self.shapeB_head(z)

        return recon, rel_logits, scale_logits, shapeR_logits, shapeB_logits, z


# ------------------------
# Training
# ------------------------

def train_model():
    scenes, rel_labels, scale_labels, red_shapes, blue_shapes = generate_dataset()
    dataset = TensorDataset(
        scenes, rel_labels, scale_labels, red_shapes, blue_shapes
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = GeomMulti().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_recon = total_rel = total_scale = total_shapeR = total_shapeB = 0.0
        correct_rel = correct_scale = correct_shapeR = correct_shapeB = 0
        total = 0

        for batch in loader:
            x, r_lbl, s_lbl, shR_lbl, shB_lbl = batch
            x    = x.to(device)
            r_lbl= r_lbl.to(device)
            s_lbl= s_lbl.to(device)
            shR_lbl= shR_lbl.to(device)
            shB_lbl= shB_lbl.to(device)

            opt.zero_grad()
            recon, rel_logits, scale_logits, shapeR_logits, shapeB_logits, z = model(x)

            recon_loss   = bce(recon, x)
            rel_loss     = ce(rel_logits, r_lbl)
            scale_loss   = ce(scale_logits, s_lbl)
            shapeR_loss  = ce(shapeR_logits, shR_lbl)
            shapeB_loss  = ce(shapeB_logits, shB_lbl)

            loss = recon_loss + rel_loss + scale_loss + shapeR_loss + shapeB_loss
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_recon   += recon_loss.item() * x.size(0)
                total_rel     += rel_loss.item() * x.size(0)
                total_scale   += scale_loss.item() * x.size(0)
                total_shapeR  += shapeR_loss.item() * x.size(0)
                total_shapeB  += shapeB_loss.item() * x.size(0)

                pred_rel   = rel_logits.argmax(dim=1)
                pred_scale = scale_logits.argmax(dim=1)
                pred_shR   = shapeR_logits.argmax(dim=1)
                pred_shB   = shapeB_logits.argmax(dim=1)

                correct_rel     += (pred_rel == r_lbl).sum().item()
                correct_scale   += (pred_scale == s_lbl).sum().item()
                correct_shapeR  += (pred_shR == shR_lbl).sum().item()
                correct_shapeB  += (pred_shB == shB_lbl).sum().item()
                total           += x.size(0)

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch}/{EPOCHS-1} | "
                f"Recon: {total_recon/total:.5f} | "
                f"RelCE: {total_rel/total:.5f} | "
                f"ScaleCE: {total_scale/total:.5f} | "
                f"ShapeRCE: {total_shapeR/total:.5f} | "
                f"ShapeBCE: {total_shapeB/total:.5f} | "
                f"Acc_rel: {100*correct_rel/total:.2f}% | "
                f"Acc_scale: {100*correct_scale/total:.2f}% | "
                f"Acc_shapeR: {100*correct_shapeR/total:.2f}% | "
                f"Acc_shapeB: {100*correct_shapeB/total:.2f}%"
            )

    # Final dataset on device for analysis
    scenes_d      = scenes.to(device)
    rel_labels_d  = rel_labels.to(device)
    scale_labels_d= scale_labels.to(device)
    red_shapes_d  = red_shapes.to(device)
    blue_shapes_d = blue_shapes.to(device)

    return model, scenes_d, rel_labels_d, scale_labels_d, red_shapes_d, blue_shapes_d


# ------------------------
# Analysis: mean latents, concept vectors, latent edits
# ------------------------

def compute_means(model, scenes, rel_labels, scale_labels, red_shapes, blue_shapes):
    model.eval()
    with torch.no_grad():
        z_all = []
        for i in range(0, scenes.size(0), BATCH_SIZE):
            batch = scenes[i:i+BATCH_SIZE]
            z = model.encoder(batch)
            z_all.append(z)
        z_all = torch.cat(z_all, dim=0)

    means_rel   = {}
    means_scale = {}
    means_red   = {}
    means_blue  = {}

    for idx, name in enumerate(RELATIONS):
        mask = (rel_labels == idx)
        if mask.any():
            means_rel[name] = z_all[mask].mean(dim=0)
    for idx, name in enumerate(SCALES):
        mask = (scale_labels == idx)
        if mask.any():
            means_scale[name] = z_all[mask].mean(dim=0)
    for idx, name in enumerate(SHAPES):
        mask = (red_shapes == idx)
        if mask.any():
            means_red[name] = z_all[mask].mean(dim=0)
    for idx, name in enumerate(SHAPES):
        mask = (blue_shapes == idx)
        if mask.any():
            means_blue[name] = z_all[mask].mean(dim=0)

    print("\nMean latent per relation:")
    for k, v in means_rel.items():
        print(f"  {k}:")
        print(f"    {v.cpu().numpy()}")

    print("\nPairwise L2 distances between relation means:")
    rel_names = list(means_rel.keys())
    for i in range(len(rel_names)):
        for j in range(i+1, len(rel_names)):
            di = torch.norm(means_rel[rel_names[i]] - means_rel[rel_names[j]]).item()
            print(f"  {rel_names[i]} <-> {rel_names[j]}: {di:.4f}")

    print("\nMean latent per scale label:")
    for k, v in means_scale.items():
        print(f"  {k}:")
        print(f"    {v.cpu().numpy()}")

    print("\nPairwise L2 distances between scale means:")
    sc_names = list(means_scale.keys())
    for i in range(len(sc_names)):
        for j in range(i+1, len(sc_names)):
            di = torch.norm(means_scale[sc_names[i]] - means_scale[sc_names[j]]).item()
            print(f"  {sc_names[i]} <-> {sc_names[j]}: {di:.4f}")

    print("\nMean latent per RED shape:")
    for k, v in means_red.items():
        print(f"  red_{k}:")
        print(f"    {v.cpu().numpy()}")

    print("\nMean latent per BLUE shape:")
    for k, v in means_blue.items():
        print(f"  blue_{k}:")
        print(f"    {v.cpu().numpy()}")

    return means_rel, means_scale, means_red, means_blue


def show_scene_grid(images, titles, path):
    """
    images: list of (2, H, W) tensors or numpy
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img
        # combine channels as RGB-ish: red in R, blue in B
        h, w = img_np.shape[1], img_np.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = img_np[0]  # red
        if img_np.shape[0] > 1:
            rgb[..., 2] = img_np[1]  # blue
        ax.imshow(rgb, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def latent_edit_experiments(model,
                            scenes,
                            rel_labels,
                            scale_labels,
                            red_shapes,
                            blue_shapes,
                            means_rel,
                            means_scale,
                            means_red,
                            means_blue):

    model.eval()
    out_dir = "outputs_latent_edit"
    os.makedirs(out_dir, exist_ok=True)

    # Precompute all latents
    with torch.no_grad():
        z_all = []
        for i in range(0, scenes.size(0), BATCH_SIZE):
            z = model.encoder(scenes[i:i+BATCH_SIZE])
            z_all.append(z)
        z_all = torch.cat(z_all, dim=0)

    # Concept directions
    v_rel_lr = means_rel["right_of"] - means_rel["left_of"]
    v_scale  = means_scale["red_larger"] - means_scale["red_smaller"]
    v_shapeR = means_red["square"] - means_red["circle"]

    # Helper: classify from latent only
    def classify_from_z(z):
        with torch.no_grad():
            rel_logits   = model.rel_head(z)
            scale_logits = model.scale_head(z)
            shapeR_logits= model.shapeR_head(z)
            shapeB_logits= model.shapeB_head(z)
            rel   = rel_logits.argmax(dim=1)
            scale = scale_logits.argmax(dim=1)
            shR   = shapeR_logits.argmax(dim=1)
            shB   = shapeB_logits.argmax(dim=1)
        return rel, scale, shR, shB

    # ------- A: relation edit: left_of -> right_of -------
    mask_rel = (rel_labels == RELATIONS.index("left_of"))
    idxs_rel = torch.nonzero(mask_rel).squeeze().cpu().numpy()
    if idxs_rel.size == 0:
        print("No left_of samples found for relation edit.")
    else:
        for k, idx in enumerate(idxs_rel[:4]):  # up to 4 examples
            idx = int(idx)
            x = scenes[idx:idx+1]
            z = z_all[idx:idx+1]

            rel0, sc0, shR0, shB0 = classify_from_z(z)

            for alpha in [0.0, 0.5, 1.0]:
                z_edit = z + alpha * v_rel_lr.unsqueeze(0)
                x_edit = model.decoder(z_edit)
                rel1, sc1, shR1, shB1 = classify_from_z(z_edit)

                imgs  = [x[0].cpu(), x_edit[0].cpu()]
                titles= [
                    f"orig: {RELATIONS[rel0.item()]}, {SCALES[sc0.item()]}",
                    f"alpha={alpha:.1f}: {RELATIONS[rel1.item()]}, {SCALES[sc1.item()]}",
                ]
                path = os.path.join(out_dir, f"edit_rel_{k}_a{int(alpha*10)}.png")
                show_scene_grid(imgs, titles, path)

    # ------- B: scale edit: red_smaller -> red_larger -------
    mask_scale = (scale_labels == SCALES.index("red_smaller"))
    idxs_sc = torch.nonzero(mask_scale).squeeze().cpu().numpy()
    if idxs_sc.size == 0:
        print("No red_smaller samples found for scale edit.")
    else:
        for k, idx in enumerate(idxs_sc[:4]):
            idx = int(idx)
            x = scenes[idx:idx+1]
            z = z_all[idx:idx+1]
            rel0, sc0, shR0, shB0 = classify_from_z(z)

            for alpha in [0.0, 0.5, 1.0]:
                z_edit = z + alpha * v_scale.unsqueeze(0)
                x_edit = model.decoder(z_edit)
                rel1, sc1, shR1, shB1 = classify_from_z(z_edit)

                imgs   = [x[0].cpu(), x_edit[0].cpu()]
                titles = [
                    f"orig: {RELATIONS[rel0.item()]}, {SCALES[sc0.item()]}",
                    f"alpha={alpha:.1f}: {RELATIONS[rel1.item()]}, {SCALES[sc1.item()]}",
                ]
                path = os.path.join(out_dir, f"edit_scale_{k}_a{int(alpha*10)}.png")
                show_scene_grid(imgs, titles, path)

    # ------- C: red shape edit: red_circle -> red_square -------
    mask_shapeR = (red_shapes == SHAPES.index("circle"))
    idxs_shR = torch.nonzero(mask_shapeR).squeeze().cpu().numpy()
    if idxs_shR.size == 0:
        print("No red_circle samples found for shape edit.")
    else:
        for k, idx in enumerate(idxs_shR[:4]):
            idx = int(idx)
            x = scenes[idx:idx+1]
            z = z_all[idx:idx+1]
            rel0, sc0, shR0, shB0 = classify_from_z(z)

            for alpha in [0.0, 0.5, 1.0]:
                z_edit = z + alpha * v_shapeR.unsqueeze(0)
                x_edit = model.decoder(z_edit)
                rel1, sc1, shR1, shB1 = classify_from_z(z_edit)

                imgs   = [x[0].cpu(), x_edit[0].cpu()]
                titles = [
                    f"orig: red_{SHAPES[shR0.item()]}, {RELATIONS[rel0.item()]}",
                    f"alpha={alpha:.1f}: red_{SHAPES[shR1.item()]}, {RELATIONS[rel1.item()]}",
                ]
                path = os.path.join(out_dir, f"edit_shape_red_{k}_a{int(alpha*10)}.png")
                show_scene_grid(imgs, titles, path)


def main():
    model, scenes, rel_labels, scale_labels, red_shapes, blue_shapes = train_model()
    means_rel, means_scale, means_red, means_blue = compute_means(
        model, scenes, rel_labels, scale_labels, red_shapes, blue_shapes
    )
    latent_edit_experiments(
        model,
        scenes,
        rel_labels,
        scale_labels,
        red_shapes,
        blue_shapes,
        means_rel,
        means_scale,
        means_red,
        means_blue,
    )

if __name__ == "__main__":
    main()
