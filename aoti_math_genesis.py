#!/usr/bin/env python3
# AOTI: GENESIS OF MATH
# The AI starts with meaningless tokens (0-99).
# It must DISCOVER that they represent quantities by arranging them in vector space
# such that Geometric Addition (Vector A + Vector B) equals Semantic Addition (Token C).

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys

# --- CONFIGURATION ---
VOCAB_SIZE = 100  # The AI knows symbols "0" through "99"
DIMENSIONS = 2    # We force it to map math to a 2D plane (X, Y)
TRAIN_SIZE = 5000 # Number of addition examples to show

class NumberManifold(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # The "Mind": A lookup table mapping each symbol to a vector.
        # Initially random. "1" might be at [0.5, -0.9], "2" at [-0.1, 0.2].
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, index_a, index_b):
        # 1. Retrieve the vector for symbol A
        vec_a = self.embeddings(index_a)
        
        # 2. Retrieve the vector for symbol B
        vec_b = self.embeddings(index_b)
        
        # 3. THE REASONING (Pure Geometry)
        # We hypothesize that "Addition" is just "Movement in Space".
        # Result Vector = Vector A + Vector B
        vec_sum = vec_a + vec_b
        
        return vec_sum

    def get_vector(self, idx):
        return self.embeddings(torch.tensor([idx])).detach().numpy()[0]

    def find_nearest_symbol(self, vector):
        # Given a coordinate, find which symbol lives closest to it.
        all_vecs = self.embeddings.weight.detach()
        # Euclidean distance from query vector to all known symbol vectors
        dists = torch.norm(all_vecs - vector, dim=1)
        nearest_idx = torch.argmin(dists).item()
        return nearest_idx, dists[nearest_idx].item()

def generate_training_data():
    # We generate "Facts" from the universe (e.g. physics observations).
    # The AI sees: (Token 5, Token 3) -> Result Token 8
    inputs = []
    targets = []
    
    for _ in range(TRAIN_SIZE):
        a = random.randint(0, 49)
        b = random.randint(0, 49)
        c = a + b
        if c < VOCAB_SIZE:
            inputs.append([a, b])
            targets.append(c)
            
    return torch.tensor(inputs), torch.tensor(targets)

def train_genesis():
    print("--- AOTI: ARCHITECTURE OF THE INFINITE ---")
    print("Task: Invent Arithmetic from Geometry.")
    print(f"Vocab: {VOCAB_SIZE} symbols | Dimensions: {DIMENSIONS}D\n")
    
    model = NumberManifold(VOCAB_SIZE, DIMENSIONS)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # We use MSE Loss to force the vectors to align
    # We want: Vector(A) + Vector(B) ≈ Vector(C)
    
    X_train, Y_train = generate_training_data()
    
    print("Initial State (Random):")
    v1 = model.get_vector(1); v2 = model.get_vector(2); v3 = model.get_vector(3)
    print(f"Vec(1): {v1} | Vec(2): {v2} | Vec(1)+Vec(2) = {v1+v2}")
    print(f"Target Vec(3): {v3} (Distance: {np.linalg.norm((v1+v2)-v3):.4f})\n")
    print("Training...")

    for epoch in range(1001):
        optimizer.zero_grad()
        
        # 1. Get the predicted vector location for (a+b)
        pred_vectors = model(X_train[:, 0], X_train[:, 1])
        
        # 2. Get the ACTUAL vector location for (c)
        # (The model is learning WHERE to place C so this is true)
        target_vectors = model.embeddings(Y_train)
        
        # 3. Minimize the distance between (Vec A + Vec B) and (Vec C)
        loss = torch.mean((pred_vectors - target_vectors) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Alignment Error {loss.item():.6f}")

    print("\n--- GENESIS COMPLETE ---")
    return model

def test_reasoning(model):
    print("\nTesting Geometric Reasoning (Vector Arithmetic):")
    
    test_cases = [(2, 2), (5, 5), (10, 20), (1, 40), (0, 50)]
    
    for a, b in test_cases:
        # 1. Get vectors
        va = model.get_vector(a)
        vb = model.get_vector(b)
        
        # 2. Perform Math (Geometry)
        v_result = va + vb
        
        # 3. Decode (Who lives at this location?)
        pred, dist = model.find_nearest_symbol(torch.tensor(v_result))
        
        print(f"Q: {a} + {b}?")
        print(f"   Vec({a}) + Vec({b}) -> Coordinate {v_result}")
        print(f"   Nearest Symbol: '{pred}' (Dist: {dist:.4f})")
        
        if pred == (a + b):
            print("   >> CORRECT. Logic verified.")
        else:
            print("   >> FAILED.")

    # VISUALIZATION OF THE DISCOVERED STRUCTURE
    print("\n--- VISUALIZING THE MANIFOLD ---")
    print("If successful, the numbers should line up in a straight line or grid.")
    print("Symbol | Vector Coordinate")
    print("-------|------------------")
    for i in range(0, 20): # Show first 20
        v = model.get_vector(i)
        print(f"  {i:02d}   | [{v[0]:.2f}, {v[1]:.2f}]")

if __name__ == "__main__":
    ai_brain = train_genesis()
    test_reasoning(ai_brain)