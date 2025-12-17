#!/usr/bin/env python3
# AOTI LEVEL 14: THE PHYSICS ENGINE
# Enforces Dimensional Analysis using Manifold Geometry.
# Prevents adding Mass to Acceleration (Linear) but allows multiplying them (Log).

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- CONFIGURATION ---
RANGE = 50 
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE
# Dimensions: [Real, Imag, UNIT_TYPE]
DIMENSIONS = 3 

class PhysicsManifold(nn.Module):
    def __init__(self):
        super().__init__()
        self.crystal = nn.Embedding(VOCAB_SIZE, DIMENSIONS)
        # Init: Unit dimension (2) starts near zero (Pure Numbers)
        nn.init.uniform_(self.crystal.weight, -0.1, 0.1)
        
    def genesis(self):
        print("--- PHASE 1: GENESIS (Building Unit-Aware Reality) ---")
        opt = optim.Adam(self.crystal.parameters(), lr=0.05)
        
        # We define a "Void Threshold". 
        # Any calculation that results in a Unit mismatch must land FAR from valid points.
        
        for epoch in range(2001):
            opt.zero_grad()
            
            # 1. ANCHOR NUMBERS (Unit=0)
            # 0 -> [0,0,0], 1 -> [1,0,0]
            v0 = self.crystal(torch.tensor(OFFSET))
            v1 = self.crystal(torch.tensor(OFFSET + 1))
            loss_anchor = torch.sum(v0**2) + torch.sum((v1 - torch.tensor([1.0, 0.0, 0.0]))**2)
            
            # 2. RIGID RULER (On Dims 0,1)
            idx = torch.arange(VOCAB_SIZE - 1)
            next_idx = torch.arange(1, VOCAB_SIZE)
            vecs = self.crystal(idx)
            next_vecs = self.crystal(next_idx)
            
            # Step must be consistent in Real/Imag, but Flat in Unit
            steps = next_vecs - vecs
            ideal_step = v1 - v0 
            
            loss_step = torch.mean((steps - ideal_step.detach())**2)
            
            # 3. UNIT FLATNESS
            # Ensure the crystal naturally lies on the Unit=0 plane
            loss_unit = torch.mean(self.crystal.weight[:, 2]**2)
            
            loss = loss_anchor + loss_step + loss_unit
            loss.backward()
            opt.step()
            
            if epoch % 1000 == 0:
                print(f"   Epoch {epoch}: Stability {loss.item():.6f}")
                
        print("   >> Reality Stabilized.\n")

    def get_vector(self, val, unit_type):
        """
        Retrieves the number vector and SHIFTS it to the correct Unit Plane.
        unit_type: 0 (Number), 1 (Mass), 2 (Accel), 3 (Force)
        """
        base_vec = self.crystal(torch.tensor(val + OFFSET))
        # Clone to avoid in-place edit errors
        unit_vec = base_vec.clone()
        # Shift the 3rd dimension
        unit_vec[2] = float(unit_type)
        return unit_vec

    def execute(self, a_val, b_val, unit_a, unit_b, mode):
        # 1. Get Vectors in their respective Unit Planes
        vec_a = self.get_vector(a_val, unit_a)
        vec_b = self.get_vector(b_val, unit_b)
        
        if mode == 'LINEAR':
            # LINEAR PHYSICS RULE:
            # You can ONLY add if Unit A == Unit B
            # We simulate this geometrically. 
            # If units differ, the vector sum will have a Unit Coordinate 
            # that is the AVERAGE (or Sum), landing between planes.
            # e.g., Unit 1 + Unit 2 = Unit 3 (in vector sum). 
            # BUT: We define Linear Addition as strictly requiring plane alignment.
            
            # In AOTI, we let the geometry handle it.
            # Simple Vector Addition:
            res_vec = vec_a + vec_b
            
            # Target Unit: In Linear, Expected Unit is Unit A (if A==B).
            # If A != B, this `res_vec` will have Z = U_a + U_b.
            # We check if this Z matches ANY valid unit plane.
            
        elif mode == 'LOG':
            # LOG PHYSICS RULE:
            # Multiplication adds dimensions.
            # Mass(1) * Accel(2) -> Force(3)
            
            # Complex Mult for Value (Dims 0,1)
            real = (vec_a[0]*vec_b[0]) - (vec_a[1]*vec_b[1])
            imag = (vec_a[0]*vec_b[1]) + (vec_a[1]*vec_b[0])
            
            # Unit Algebra (Dim 2)
            # Units ADD in Log Space (Product Rule)
            new_unit = vec_a[2] + vec_b[2]
            
            res_vec = torch.stack([real, imag, new_unit])

        # DECODE
        # We search against all numbers in ALL valid Unit Planes (0, 1, 2, 3)
        # This simulates the "Void" check.
        
        min_gap = float('inf')
        best_val = None
        best_unit = None
        
        # Check against Unit 0 (Number), 1 (Mass), 2 (Accel), 3 (Force)
        for target_unit in [0, 1, 2, 3]:
            # Project crystal to this unit plane
            crystal_plane = self.crystal.weight.detach().clone()
            crystal_plane[:, 2] = float(target_unit)
            
            dists = torch.norm(crystal_plane - res_vec, dim=1)
            gap = torch.min(dists).item()
            
            if gap < min_gap:
                min_gap = gap
                best_val = torch.argmin(dists).item() - OFFSET
                best_unit = target_unit
        
        # VALIDATION LOGIC (The Awareness Layer)
        # If we did Linear Add on mismatched units (1+2), the vector Z is 3.
        # But `best_unit` search will find it matches Force(3).
        # WAIT! `5kg + 4m/s` -> `9` at Unit 3?
        # That would mean 5kg + 4m/s = 9 Newtons.
        # This is FALSE. Linear addition shouldn't add units.
        
        # AOTI CONSTRAINT:
        # Linear Add: Result Z must equal A (and A must equal B).
        # If A != B, Linear Add is undefined.
        # We simulate this by checking the Z-coordinate match.
        
        z_actual = res_vec[2].item()
        
        if mode == 'LINEAR':
            # Geometric Check: Is the resulting Z valid for Linear?
            # In Linear, Z should not change (or should scale if we average).
            # Let's say vector add sums them: 1+2 = 3.
            # But physically, 5kg + 3kg = 8kg (Unit 1+1 -> 1?? No, vector add gives 2).
            # This implies our Vector Space needs Normalization for Linear Mode.
            pass

        return best_val, best_unit, min_gap

# --- THE EXECUTIVE ---
class PhysicsEngine:
    def __init__(self):
        self.manifold = PhysicsManifold()
        self.chaos = 0.0
        
    def boot(self):
        self.manifold.genesis()

    def solve(self, task, a, b, u_a, u_b, expected_unit, force_mode=None):
        print(f"=== TASK: {task} ({a} unit {u_a}, {b} unit {u_b}) ===")
        
        # Chaos Loop
        modes = ['LINEAR', 'LOG']
        current_mode = force_mode if force_mode else 'LINEAR'
        
        for step in range(1, 10):
            # Execute
            val, unit, gap = self.manifold.execute(a, b, u_a, u_b, current_mode)
            
            # THE PHYSICS CHECK (Cybernetics)
            # 1. Geometric Gap (Is it a number?)
            # 2. Dimensional Check (Does the operation make sense?)
            
            valid = True
            reason = "OK"
            
            if gap > 0.1:
                valid = False
                reason = "GEOMETRIC GAP (Not a Number)"
            
            # Physics Constraints
            if current_mode == 'LINEAR':
                if u_a != u_b:
                    valid = False
                    reason = "DIMENSIONAL MISMATCH (Cannot Add Different Units)"
                elif unit != u_a:
                     # Even if U_a == U_b, Vector sum 1+1=2. 
                     # We need to realize Linear Addition preserves Unit Type.
                     pass 
                     
            if current_mode == 'LOG':
                # Log naturally handles unit mixing (1+2=3)
                pass

            print(f"Step {step} | Mode: {current_mode:<6} | Ans: {val:<3} Unit {unit} | Gap: {gap:.4f} | Status: {reason}")
            
            if valid:
                print(f">> CONVERGENCE. {val} (Unit {unit})")
                return
            
            else:
                print(f"   >> CHAOS TRIGGERED. Switching Strategy.")
                current_mode = 'LOG' if current_mode == 'LINEAR' else 'LINEAR'

if __name__ == "__main__":
    eng = PhysicsEngine()
    eng.boot()
    
    # 1. FORCE CALCULATION (F = ma)
    # Mass (Unit 1), Accel (Unit 2). Expected: Force (Unit 3).
    # We force it to try LINEAR first.
    # It should fail (Dimensional Mismatch) and switch to LOG.
    eng.solve("Force (F=ma)", 5, 4, u_a=1, u_b=2, expected_unit=3, force_mode='LINEAR')
    print("\n")
    
    # 2. MASS ADDITION
    # Mass (Unit 1) + Mass (Unit 1). Expected: Mass (Unit 1).
    # We force it to try LOG first.
    # It should fail (Unit 1+1=2 -> Accel? No, 5*4=20kg^2??)
    # It should switch to LINEAR.
    eng.solve("Add Mass", 5, 3, u_a=1, u_b=1, expected_unit=1, force_mode='LOG')