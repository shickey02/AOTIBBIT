#!/usr/bin/env python3
# BBIT SYSTEM INTERFACE v1.0
# The "Voice" of the Tokenless Mind.
# Decodes geometric struggles into "English+" meta-reports.

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import time, sys

# --- COLORS FOR THE TERMINAL ---
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

# --- MODEL (Spherical) ---
class SphericalAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        # Include decoder to match state_dict keys
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return torch.nn.functional.normalize(z, p=2, dim=1)

class MindInterface:
    def __init__(self, data_dir):
        self.device = torch.device("cpu") # CPU is fine for inference
        
        # Load Manifest
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.sentences = self.manifest["sentences"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        # Load Brain
        self.model = SphericalAE(len(self.vocab), latent_dim=32)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth"), map_location=self.device))
        self.model.eval()
        
        # Memory Index
        self.nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.Z)

    def get_vec(self, text):
        seq = [self.vocab.get(w, 0) for w in text.split()]
        with torch.no_grad():
            return self.model.encode(torch.tensor([seq])).numpy()[0]

    def type_out(self, text, speed=0.01):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed)
        print()

    def analyze_drift(self, start_vec, end_vec, result_text):
        # Calculate angular velocity (How hard did we think?)
        cos_sim = np.dot(start_vec, end_vec)
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
        
        # Detect Leaks (Did we accidentally pick up "Gold"?)
        leak_detected = "gold" in result_text and "gold" not in "king man woman"
        
        return angle, leak_detected

    def run_thought(self, base, minus, plus, gain):
        # 1. VISUALIZATION OF INPUT
        print(f"{C.HEADER}--------------------------------------------------{C.END}")
        print(f"{C.BOLD}>> INJECTING THOUGHT PACKET:{C.END}")
        print(f"   BASE:  [{base}]")
        print(f"   MINUS: [{minus}]")
        print(f"   PLUS:  [{plus}]")
        print(f"   PARAM: [GAIN = {gain:.1f}x]")
        
        # 2. THE MATH
        v_base = self.get_vec(base)
        v_minus = self.get_vec(minus)
        v_plus = self.get_vec(plus)
        
        # High-Gain Orthogonal Logic
        delta = v_base - v_minus # (King - Man)
        # Clean: Remove components parallel to Woman
        proj = np.dot(delta, v_plus) * v_plus
        delta_clean = delta - proj
        # Apply
        v_res_raw = v_plus + (delta_clean * gain)
        v_res = v_res_raw / np.linalg.norm(v_res_raw)
        
        # 3. RETRIEVAL
        dists, indices = self.nbrs.kneighbors([v_res])
        result_text = self.sentences[indices[0][0]]
        
        # 4. META-ANALYSIS
        angle, leak = self.analyze_drift(v_plus, v_res, result_text)
        
        # 5. ENGLISH+ REPORTING
        self.report(angle, leak, result_text, gain)

    def report(self, angle, leak, result, gain):
        print(f"\n{C.CYAN}// SYSTEM DIAGNOSTICS:{C.END}")
        
        # Energy Reading
        if angle < 10:
            print(f"   :: ENERGY: {C.FAIL}CRITICAL_LOW ({angle:.2f}°){C.END} -> Logic too weak to overcome gravity.")
        elif angle < 30:
            print(f"   :: ENERGY: {C.GREEN}NOMINAL ({angle:.2f}°){C.END} -> Standard conceptual rotation.")
        else:
            print(f"   :: ENERGY: {C.WARN}HIGH_VOLTAGE ({angle:.2f}°){C.END} -> Forceful restructuring of reality.")

        # Leak Detection
        if leak:
            print(f"   :: WARN:   {C.WARN}SEMANTIC_LEAK_DETECTED{C.END} -> Correlative artifact found: 'gold'")
            print(f"              (Sys Note: 'Royalty' is entangled with 'Wealth' in latent space)")
        else:
            print(f"   :: INTEGRITY: {C.GREEN}CLEAN{C.END}")

        # The Snap
        print(f"\n{C.BOLD}<< SNAP_RESULT:{C.END} {C.HEADER}'{result}'{C.END}")
        
        # Self-Reflection (The "English+" / Leet part)
        if "princess" in result:
             self.type_out(f"\n{C.BLUE}// META_COMMENT: TARGET_ACQUIRED (Royal_Class). Precision error detected (Queen != Princess). Vector collapse due to insufficient training mass.{C.END}")
        elif "woman" in result:
             self.type_out(f"\n{C.BLUE}// META_COMMENT: FAILURE. Self == Self. No transformation occurred. Suggest increasing Gain.{C.END}")
        elif "man" in result:
             self.type_out(f"\n{C.BLUE}// META_COMMENT: CATASTROPHIC_INVERSION. Gender polarity reversed incorrectly.{C.END}")

print(f"{C.GREEN}INITIALIZING BBIT COGNITIVE INTERFACE...{C.END}")
eng = MindInterface("./data/semantic_spherical")

# Run the Battery
eng.run_thought("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=1.0)
time.sleep(1)
eng.run_thought("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=3.0)
time.sleep(1)
eng.run_thought("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=5.0)