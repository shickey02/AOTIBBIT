#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase9.py
#
# Phase 9: NO TRAINING
# - Phase 7 provides representation + factor heads
# - Phase 8 provides calibrated thresholds (Tb, To)
# - Phase 9 just standardizes/export derived discrete labels for downstream geometry work

import os

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"
PHASE8_THRESH = "outputs_edges_relternary256_phase8/phase8_thresholds.json"

print("[train64-ternary-phase9-256] Phase 9 has no training step.")
print("[train64-ternary-phase9-256] Using Phase 7 model + Phase 8 calibrated thresholds.")

assert os.path.exists(PHASE7_CKPT), f"Phase 7 checkpoint not found: {PHASE7_CKPT}"
assert os.path.exists(PHASE8_THRESH), f"Phase 8 thresholds not found: {PHASE8_THRESH}"

print(f"[train64-ternary-phase9-256] OK -> {PHASE7_CKPT}")
print(f"[train64-ternary-phase9-256] OK -> {PHASE8_THRESH}")
