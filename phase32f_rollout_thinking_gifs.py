#!/usr/bin/env python3
# phase32f_rollout_thinking_gifs.py
#
# Encode -> apply Δxyz operator in latent -> decode -> GIF.
#
# Key idea:
#   Fit linear W: Z -> (dx,dy,dz)  (D x 3)
#   Then latent operator for desired delta_xyz:
#       delta_z = pinv(W) @ delta_xyz   (D,)
#
# Exports:
#   outdir/
#     gifs/*.gif
#     rollouts.csv
#
# Notes:
# - Works best with your finetuned model checkpoint.
# - If decoder path can't be found, script errors with helpful prints.

import os, argparse, csv, importlib.util
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont

# ---------------- utils ----------------

def ridge_fit(X, Y, lam=1e-2):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    D = X.shape[1]
    A = X.T @ X
    A.flat[::D+1] += lam
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W  # [D,K]

def try_load_py_class(py_path, class_name):
    spec = importlib.util.spec_from_file_location("user_model", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    cls = getattr(mod, class_name)
    return cls

def load_ckpt_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Common patterns
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
            # merge (if you ever save split dicts)
            sd = {}
            sd.update(ckpt["encoder_state_dict"])
            sd.update(ckpt["decoder_state_dict"])
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # maybe it IS the raw state_dict already
            sd = ckpt
        else:
            raise ValueError(
                f"Checkpoint dict format not understood. Keys: {list(ckpt.keys())[:50]}"
            )
    else:
        # full model object
        if hasattr(ckpt, "state_dict"):
            sd = ckpt.state_dict()
        else:
            raise ValueError(f"Checkpoint type not understood: {type(ckpt)}")

    miss, unexp = model.load_state_dict(sd, strict=False)
    return miss, unexp


@torch.no_grad()
def forward_model(model, x):
    """
    Your ConvAEHeads forward has varied return shapes across phases.
    We attempt to detect reconstruction + latent.
    Returns: recon (B,3,64,64), z (B,D)
    """
    out = model(x)
    # Common patterns:
    # - out may be dict
    # - out may be tuple with recon first
    if isinstance(out, dict):
        recon = out.get("recon", None) or out.get("x_hat", None)
        z = out.get("z", None) or out.get("latent", None)
        if recon is None or z is None:
            raise ValueError(f"Forward dict missing recon/z keys. Found keys={list(out.keys())}")
        return recon, z
    if isinstance(out, (tuple, list)):
        # guess: first is recon, second is z
        if len(out) < 2:
            raise ValueError("Forward returned tuple/list too short.")
        recon = out[0]
        z = out[1]
        # if model returns (z, recon), swap if needed
        if recon.ndim == 2 and z.ndim == 4:
            recon, z = z, recon
        return recon, z
    raise ValueError(f"Forward output type not recognized: {type(out)}")

@torch.no_grad()
def encode_only(model, x):
    # Prefer explicit encoder method if exists
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        z = model.encode(x)
        return z
    # else: run forward and grab z
    _, z = forward_model(model, x)
    return z

@torch.no_grad()
def decode_only(model, z):
    # Prefer explicit decode method if exists
    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        xh = model.decode(z)
        return xh
    # else try decoder attribute
    if hasattr(model, "decoder") and callable(getattr(model, "decoder")):
        return model.decoder(z)
    raise ValueError("Could not find a decode path (model.decode or model.decoder).")

def to_uint8_img(x):
    """
    x: torch tensor [3,64,64] in 0..1 or -1..1-ish
    """
    x = x.detach().float().cpu().numpy()
    # robust normalize
    # if looks like -1..1
    if x.min() < -0.2:
        x = (x + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1,2,0))  # HWC
    return x

def render_frame(img_u8, text=None, scale=8, pad=18, dark=True):
    """
    Dark mode output with minimal HUD.
    """
    im = Image.fromarray(img_u8, mode="RGB").resize((img_u8.shape[1]*scale, img_u8.shape[0]*scale), resample=Image.NEAREST)
    W, H = im.size
    canvas = Image.new("RGB", (W, H + pad), (0,0,0) if dark else (255,255,255))
    canvas.paste(im, (0,0))
    if text:
        draw = ImageDraw.Draw(canvas)
        # default PIL font
        draw.text((8, H+2), text, fill=(220,220,220) if dark else (30,30,30))
    return canvas

def save_gif(frames_pil, out_path, duration_ms=90):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="images.npy [N,3,64,64] float32")
    ap.add_argument("--labels", required=True, help="labels.npz from phase32a")
    ap.add_argument("--encoder_py", required=True)
    ap.add_argument("--encoder_class", required=True)
    ap.add_argument("--ckpt", required=True, help="phase7 or finetuned_model.pt")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--num_seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--step_scale", type=float, default=0.35, help="multiplier on delta_z per step")
    ap.add_argument("--ridge_lam", type=float, default=1e-2)

    ap.add_argument("--gif_ms", type=int, default=90)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--scale", type=int, default=8)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gif_dir = os.path.join(args.outdir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print("[device]", device)

    # Load data
    X = np.load(args.images).astype(np.float32)  # [N,3,64,64]
    lab = np.load(args.labels)
    dx = lab["dx"].astype(np.float32).reshape(-1)
    dy = lab["dy"].astype(np.float32).reshape(-1)
    dz = lab["dz"].astype(np.float32).reshape(-1)
    dxyz = np.stack([dx,dy,dz], axis=1).astype(np.float32)

    N = X.shape[0]
    print("[data]", X.shape, "labels", dxyz.shape)

    # Build model
    ModelCls = try_load_py_class(args.encoder_py, args.encoder_class)
    model = ModelCls().to(device)
    miss, unexp = load_ckpt_into_model(model, args.ckpt, device)
    print("[load] missing:", len(miss), "unexpected:", len(unexp))
    model.eval()

    # Encode all latents (for fitting W) using batches
    # We only need enough samples for a solid ridge fit; use up to 20000.
    fitN = min(20000, N)
    fit_idx = rng.choice(N, size=fitN, replace=False)
    bs = 512
    Zfit = []
    for i in range(0, fitN, bs):
        j = fit_idx[i:i+bs]
        xb = torch.from_numpy(X[j]).to(device)
        zb = encode_only(model, xb).detach().float().cpu().numpy()
        Zfit.append(zb)
        if (i//bs) % 10 == 0:
            print(f"[enc-fit] {i}/{fitN}")
    Zfit = np.concatenate(Zfit, axis=0).astype(np.float32)
    print("[enc-fit] Zfit:", Zfit.shape)

    # Fit W: Z -> dxyz
    W = ridge_fit(Zfit.astype(np.float64), dxyz[fit_idx].astype(np.float64), lam=args.ridge_lam)  # [D,3]
    # Pseudoinverse to map desired Δxyz -> Δz
    W_pinv = np.linalg.pinv(W)  # [3,D]

    D = Zfit.shape[1]
    print("[fit] W:", W.shape, "pinv:", W_pinv.shape, "latent D:", D)

    # Choose seed images: try to pick diverse dz or just random
    seed_idx = rng.choice(N, size=args.num_seeds, replace=False)

    # Define a set of “thinking events” (Δxyz patterns)
    # You can expand this list whenever.
    patterns = [
        ("push_x",      np.array([+1.0,  0.0,  0.0], dtype=np.float64)),
        ("pull_x",      np.array([-1.0,  0.0,  0.0], dtype=np.float64)),
        ("push_y",      np.array([ 0.0, +1.0,  0.0], dtype=np.float64)),
        ("pull_y",      np.array([ 0.0, -1.0,  0.0], dtype=np.float64)),
        ("push_z",      np.array([ 0.0,  0.0, +1.0], dtype=np.float64)),
        ("pull_z",      np.array([ 0.0,  0.0, -1.0], dtype=np.float64)),
        ("diag_xyz",    np.array([+1.0, +1.0, +1.0], dtype=np.float64) / np.sqrt(3.0)),
        ("diag_x_y",    np.array([+1.0, +1.0,  0.0], dtype=np.float64) / np.sqrt(2.0)),
    ]

    # CSV log
    csv_path = os.path.join(args.outdir, "rollouts.csv")
    with open(csv_path, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["gif", "seed_idx", "pattern", "t", "pred_dx", "pred_dy", "pred_dz"])

        # Run rollouts
        for si, sidx in enumerate(seed_idx):
            xb0 = torch.from_numpy(X[sidx:sidx+1]).to(device)
            z0 = encode_only(model, xb0).detach().float().cpu().numpy().reshape(-1)  # [D]

            for (pname, delta_xyz_unit) in patterns:
                # Convert desired Δxyz direction into latent Δz direction
                delta_z = (W_pinv.T @ delta_xyz_unit).astype(np.float64)  # [D]
                # Normalize and scale
                nz = np.linalg.norm(delta_z) + 1e-12
                delta_z = delta_z / nz

                frames = []
                for t in range(args.steps):
                    zt = z0.astype(np.float64) + (t * args.step_scale) * delta_z
                    zt_t = torch.from_numpy(zt.astype(np.float32)).to(device).unsqueeze(0)

                    # decode
                    xh = decode_only(model, zt_t)[0]  # [3,64,64]
                    img_u8 = to_uint8_img(xh)

                    # predicted xyz from W
                    pred = (zt @ W).astype(np.float64)  # [3]
                    hud = None
                    if True:
                        hud = f"{pname}  t={t:02d}  xyz=({pred[0]:+.2f},{pred[1]:+.2f},{pred[2]:+.2f})"

                    frame = render_frame(img_u8, text=hud, scale=args.scale, pad=18, dark=args.dark)
                    frames.append(frame)

                    wcsv.writerow([f"seed{si:02d}_{pname}.gif", int(sidx), pname, int(t), float(pred[0]), float(pred[1]), float(pred[2])])

                out_gif = os.path.join(gif_dir, f"seed{si:02d}_{pname}.gif")
                save_gif(frames, out_gif, duration_ms=args.gif_ms)
                print("[gif]", out_gif)

    print("[ok] wrote:", csv_path)
    print("[ok] gifs in:", gif_dir)

if __name__ == "__main__":
    main()
