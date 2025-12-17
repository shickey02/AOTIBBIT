#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase8.py
#
# Phase 8: NO TRAINING
# Uses Phase 7 checkpoint. Threshold calibration happens in eval.

import os
print("[train64-ternary-phase8-256] Phase 8 has no training step.")
print("[train64-ternary-phase8-256] Using Phase 7 model directly.")

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"
assert os.path.exists(PHASE7_CKPT), "Phase 7 checkpoint not found"

print(f"[train64-ternary-phase8-256] OK -> {PHASE7_CKPT}")
