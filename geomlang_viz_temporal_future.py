#!/usr/bin/env python3
# geomlang_viz_temporal_future.py
#
# Simple visualizations for 6.3:
#   - frame-wise Rel/Scale accuracy over epochs
#   - Future@T Rel/Scale accuracy over epochs
#
# Expects:
#   outputs_edges/temporal_future_stats.npz
#
# Run:
#   python bbit_geomlang/geomlang_viz_temporal_future.py

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR    = "outputs_edges"
STATS_PATH = os.path.join(OUT_DIR, "temporal_future_stats.npz")


def main():
    if not os.path.exists(STATS_PATH):
        print(f"[viz] Stats file not found: {STATS_PATH}")
        return

    data = np.load(STATS_PATH)

    train_loss = data["train_loss"]
    rel_acc = data["train_rel_acc"]
    scale_acc = data["train_scale_acc"]
    fut_rel_acc = data["train_future_rel_acc"]
    fut_scale_acc = data["train_future_scale_acc"]

    epochs = np.arange(1, len(train_loss) + 1)

    # 1) Frame-wise accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, rel_acc, marker="o", label="RelAcc (frame-wise)")
    plt.plot(epochs, scale_acc, marker="s", label="ScaleAcc (frame-wise)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("6.3 – Frame-wise relation/scale accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    frame_acc_path = os.path.join(OUT_DIR, "temporal_future_frame_acc.png")
    plt.tight_layout()
    plt.savefig(frame_acc_path, dpi=150)
    plt.close()
    print(f"[viz] Saved frame-wise accuracy plot -> {frame_acc_path}")

    # 2) Future@T accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, fut_rel_acc, marker="o", label="Future@T RelAcc")
    plt.plot(epochs, fut_scale_acc, marker="s", label="Future@T ScaleAcc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("6.3 – Future@T relation/scale accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    fut_acc_path = os.path.join(OUT_DIR, "temporal_future_future_acc.png")
    plt.tight_layout()
    plt.savefig(fut_acc_path, dpi=150)
    plt.close()
    print(f"[viz] Saved Future@T accuracy plot -> {fut_acc_path}")

    # 3) Loss curve (optional but handy)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("6.3 – Training loss (including future loss)")
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_path = os.path.join(OUT_DIR, "temporal_future_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[viz] Saved loss curve -> {loss_path}")


if __name__ == "__main__":
    main()
