#!/usr/bin/env python3
# phase32a_generate_3d_dataset.py
#
# Generate 2-object synthetic scenes with TRUE 3D labels.
# Renders to 64x64 RGB (or 3ch) using a simple perspective projection.
#
# Output:
#   outdir/
#     images.npy        [N, 3, H, W] float32 in [0,1]
#     labels.npz        arrays: xA,yA,zA,xB,yB,zB, scales, shapes, relations, etc.
#
# No torch required. Pure numpy. (You can later load into torch.)
#
import os, argparse
import numpy as np

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def render_soft_disk(img, cx, cy, r, color):
    # img: [H,W,3]
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx)**2 + (yy - cy)**2
    # soft edge
    sigma = max(1.0, 0.35*r)
    a = np.exp(-d2/(2.0*sigma*sigma))
    a = clamp01(a)
    for k in range(3):
        img[...,k] = img[...,k]*(1-a) + color[k]*a
    return img

def render_soft_square(img, cx, cy, r, color):
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    # soft edge around max(dx,dy) <= r
    dist = np.maximum(dx, dy)
    sigma = max(1.0, 0.35*r)
    a = np.exp(-((dist - r).clip(min=0.0)**2)/(2.0*sigma*sigma))
    a = clamp01(a)
    for k in range(3):
        img[...,k] = img[...,k]*(1-a) + color[k]*a
    return img

def project_persp(x, y, z, f=1.2):
    """
    x,y,z in [-1,1] box.
    Simple camera at z_cam = +2.2 looking toward origin.
    """
    z_cam = 2.2
    zz = (z_cam - z)  # depth > 0
    # perspective divide
    u = f * (x / zz)
    v = f * (y / zz)
    return u, v, zz

def to_pixels(u, v, H, W):
    # map u,v roughly in [-0.7,0.7] -> pixels
    px = (u * 0.65 + 0.5) * (W-1)
    py = (0.5 - v * 0.65) * (H-1)
    return px, py

def overlap_2d(pxA, pyA, rA, pxB, pyB, rB):
    d = np.sqrt((pxA-pxB)**2 + (pyA-pyB)**2)
    return float(d < (rA + rB)*0.85)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=50000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--z_bias_front", type=float, default=0.0, help="bias z toward front (negative) or back (positive)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    H, W = args.H, args.W
    X = np.zeros((args.N, 3, H, W), dtype=np.float32)

    # labels
    xA = np.zeros(args.N, np.float32); yA = np.zeros(args.N, np.float32); zA = np.zeros(args.N, np.float32)
    xB = np.zeros(args.N, np.float32); yB = np.zeros(args.N, np.float32); zB = np.zeros(args.N, np.float32)
    sA = np.zeros(args.N, np.float32); sB = np.zeros(args.N, np.float32)
    shapeA = np.zeros(args.N, np.int64); shapeB = np.zeros(args.N, np.int64)  # 0=disk,1=square

    # relations / numeric targets
    dx = np.zeros(args.N, np.float32); dy = np.zeros(args.N, np.float32); dz = np.zeros(args.N, np.float32)
    dist = np.zeros(args.N, np.float32)
    left = np.zeros(args.N, np.int64); above = np.zeros(args.N, np.int64); front = np.zeros(args.N, np.int64)
    ov2d = np.zeros(args.N, np.int64)

    for i in range(args.N):
        # sample 3D positions
        xa, ya, za = rng.uniform(-1, 1, size=3)
        xb, yb, zb = rng.uniform(-1, 1, size=3)
        za += args.z_bias_front * 0.25
        zb += args.z_bias_front * 0.25
        za = float(np.clip(za, -1, 1))
        zb = float(np.clip(zb, -1, 1))

        # scales
        sa = float(rng.uniform(0.10, 0.28))
        sb = float(rng.uniform(0.10, 0.28))

        # shapes
        shA = int(rng.integers(0, 2))
        shB = int(rng.integers(0, 2))

        # project
        ua, va, da = project_persp(xa, ya, za)
        ub, vb, db = project_persp(xb, yb, zb)
        pxa, pya = to_pixels(ua, va, H, W)
        pxb, pyb = to_pixels(ub, vb, H, W)

        # radius in pixels: scale / depth
        rA = max(2.0, (sa / da) * W * 0.85)
        rB = max(2.0, (sb / db) * W * 0.85)

        img = np.zeros((H, W, 3), dtype=np.float32)

        # painter's order: draw farther first
        if da > db:
            order = [(pxa,pya,rA,shA,(1.0,0.2,0.2)), (pxb,pyb,rB,shB,(0.2,0.4,1.0))]
        else:
            order = [(pxb,pyb,rB,shB,(0.2,0.4,1.0)), (pxa,pya,rA,shA,(1.0,0.2,0.2))]

        for (px,py,r,sh,col) in order:
            if sh == 0:
                img = render_soft_disk(img, px, py, r, col)
            else:
                img = render_soft_square(img, px, py, r, col)

        # store
        X[i] = np.transpose(img, (2,0,1))

        xA[i], yA[i], zA[i] = xa, ya, za
        xB[i], yB[i], zB[i] = xb, yb, zb
        sA[i], sB[i] = sa, sb
        shapeA[i], shapeB[i] = shA, shB

        dx[i], dy[i], dz[i] = (xb-xa), (yb-ya), (zb-za)
        dist[i] = float(np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2))

        left[i]  = int(xb > xa)      # B right of A => left=1 means "B is right"
        above[i] = int(yb > ya)
        front[i] = int(zb < za)      # more negative z is "front" (closer)

        ov2d[i] = int(overlap_2d(pxa,pya,rA, pxb,pyb,rB))

    np.save(os.path.join(args.outdir, "images.npy"), X)
    np.savez(
        os.path.join(args.outdir, "labels.npz"),
        xA=xA,yA=yA,zA=zA,xB=xB,yB=yB,zB=zB,
        sA=sA,sB=sB,shapeA=shapeA,shapeB=shapeB,
        dx=dx,dy=dy,dz=dz,dist=dist,left=left,above=above,front=front,ov2d=ov2d
    )

    print("[ok] wrote:", os.path.join(args.outdir, "images.npy"))
    print("[ok] wrote:", os.path.join(args.outdir, "labels.npz"))
    print("[info] N=", args.N)

if __name__ == "__main__":
    main()
