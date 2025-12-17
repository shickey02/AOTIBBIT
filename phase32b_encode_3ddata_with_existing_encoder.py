#!/usr/bin/env python3
# phase32b_encode_3ddata_with_existing_encoder.py
#
# Encodes Phase32a 3D dataset images into your EXISTING latent space.
#
# Inputs:
#   --images   path/to/images.npy   [N,3,H,W] float32 in [0,1]
#   Encoder loader (choose ONE):
#     A) TorchScript (recommended):
#        --encoder_jit path/to/encoder.pt
#     B) Python module + checkpoint:
#        --encoder_py path/to/model_def.py
#        --encoder_class ClassName
#        --ckpt path/to/checkpoint.pt
#
# Output:
#   --out_latents  latents.npy  [N, LATENT_DIM]
#
import os, argparse, importlib.util
import numpy as np
import torch

def load_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("user_encoder_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def try_get_encoder_from_model(model):
    """
    Try common patterns:
    - model.encode(x)
    - model.encoder(x)
    - model.ae.encode(x)
    - model.ae.encoder(x)
    - model.forward returns (recon, z) or dict with 'z'
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        return lambda x: model.encode(x)

    if hasattr(model, "encoder") and callable(getattr(model, "encoder")):
        return lambda x: model.encoder(x)

    if hasattr(model, "ae"):
        ae = getattr(model, "ae")
        if hasattr(ae, "encode") and callable(getattr(ae, "encode")):
            return lambda x: ae.encode(x)
        if hasattr(ae, "encoder") and callable(getattr(ae, "encoder")):
            return lambda x: ae.encoder(x)

    def _fallback(x):
        y = model(x)
        if isinstance(y, (tuple, list)) and len(y) >= 2:
            return y[-1]
        if isinstance(y, dict):
            for k in ["z", "latents", "latent", "h"]:
                if k in y:
                    return y[k]
        raise ValueError("Could not extract latents from model output. Add an encode() method or adjust fallback.")
    return _fallback

def load_encoder(args, device):
    if args.encoder_jit:
        enc = torch.jit.load(args.encoder_jit, map_location=device)
        enc.eval()
        return enc, lambda x: enc(x)

    if not (args.encoder_py and args.encoder_class and args.ckpt):
        raise ValueError("Need either --encoder_jit OR (--encoder_py --encoder_class --ckpt).")

    mod = load_module_from_path(args.encoder_py)
    if not hasattr(mod, args.encoder_class):
        raise ValueError(f"encoder_class '{args.encoder_class}' not found in {args.encoder_py}")
    cls = getattr(mod, args.encoder_class)

    model = cls()
    ckpt = torch.load(args.ckpt, map_location=device)

    # common checkpoint formats
    sd = None
    for key in ["state_dict", "model_state", "model", "net", "ae_state_dict"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    if sd is None and isinstance(ckpt, dict):
        # maybe it's directly a state_dict
        if any(k.endswith(".weight") for k in ckpt.keys()):
            sd = ckpt
    if sd is None:
        raise ValueError("Could not find a state_dict inside checkpoint. Inspect ckpt keys.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] missing:", len(missing), "unexpected:", len(unexpected))

    model.to(device).eval()
    get_z = try_get_encoder_from_model(model)
    return model, get_z

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_latents", required=True)
    ap.add_argument("--batch", type=int, default=256)

    # encoder options
    ap.add_argument("--encoder_jit", default=None)
    ap.add_argument("--encoder_py", default=None)
    ap.add_argument("--encoder_class", default=None)
    ap.add_argument("--ckpt", default=None)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("[device]", device)

    X = np.load(args.images)  # [N,3,H,W]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    N = X.shape[0]
    print("[data] images:", X.shape, X.dtype)

    model, get_z = load_encoder(args, device)

    latents = []
    for i in range(0, N, args.batch):
        xb = torch.from_numpy(X[i:i+args.batch]).to(device)
        z = get_z(xb)
        if isinstance(z, (tuple, list)):
            z = z[-1]
        z = z.detach()
        if z.ndim > 2:
            # if encoder outputs feature map, flatten
            z = torch.flatten(z, start_dim=1)
        latents.append(z.cpu().numpy().astype(np.float32))
        if (i // args.batch) % 20 == 0:
            print(f"[enc] {i}/{N}")

    Z = np.concatenate(latents, axis=0)
    os.makedirs(os.path.dirname(args.out_latents), exist_ok=True)
    np.save(args.out_latents, Z)
    print("[ok] wrote:", args.out_latents, "shape=", Z.shape)

if __name__ == "__main__":
    main()
