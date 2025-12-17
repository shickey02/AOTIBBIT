#!/usr/bin/env python3
# BBIT SEMANTIC SCULPTOR
# Uses Vector Navigation to "write" by selecting words that match a target meaning.
# Requires: pip install sentence-transformers

import random, time, copy
import numpy as np
import sys

# Try to import the encoder
try:
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = 'all-MiniLM-L6-v2' # Fast, lightweight model
    print(f"Loading Semantic Brain ({MODEL_NAME})...")
    ENCODER = SentenceTransformer(MODEL_NAME)
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

# --- 1. THE VOCABULARY (The Raw Material) ---
# A soup of random words from different genres to carve from.
VOCABULARY = [
    "dark", "light", "cyber", "forest", "steel", "soul", "run", "fast",
    "algorithm", "love", "hate", "crimson", "silent", "scream", "whisper",
    "neon", "god", "system", "failure", "hope", "shattered", "eternal",
    "code", "ancient", "future", "blood", "stars", "void", "machine",
    "flesh", "burn", "cold", "winter", "summer", "dream", "nightmare",
    "golden", "rot", "bloom", "sky", "ocean", "digital", "analog",
    "chaos", "order", "king", "slave", "freedom", "prison", "mind",
    "shadow", "dust", "galaxy", "quantum", "neural", "ghost", "shell"
]

# --- 2. THE APEX WRITER (The Mind) ---
class SemanticAgent:
    def __init__(self, target_phrase, sentence_length=5):
        # 1. Sense the Goal (Embed the target concept)
        self.target_vec = ENCODER.encode(target_phrase)
        self.target_desc = target_phrase
        
        # 2. Initialize State (Random gibberish)
        self.sentence = [random.choice(VOCABULARY) for _ in range(sentence_length)]
        
        # 3. Memory
        self.best_sentence = list(self.sentence)
        self.best_score = float('inf') # Lower distance is better
        
        # 4. Cybernetics
        self.stagnation = 0
        self.patience = 10
        self.chaos_level = 0.0
        
    def get_current_text(self):
        return " ".join(self.sentence)

    def think(self, current_vec):
        # Calculate Cosine Distance (1 - Similarity)
        # We want Distance to be 0.
        score = 1.0 - np.dot(current_vec, self.target_vec) / (
            np.linalg.norm(current_vec) * np.linalg.norm(self.target_vec)
        )
        
        if score < self.best_score:
            # Improvement!
            self.best_score = score
            self.best_sentence = list(self.sentence)
            self.stagnation = 0
            self.chaos_level = 0.0
            return True, score
        
        # Stagnation -> Anxiety
        self.stagnation += 1
        if self.stagnation > self.patience:
            self.chaos_level = min(0.8, (self.stagnation - self.patience) / 20.0)
            
        return False, score

    def act(self):
        candidate = list(self.sentence)
        
        # MODE A: PRECISION (Fine-tuning)
        if self.chaos_level < 0.2:
            # Swap ONE word for a random word in vocab
            idx = random.randint(0, len(candidate)-1)
            candidate[idx] = random.choice(VOCABULARY)
            return candidate, "Precision"

        # MODE B: BRAINSTORM (Chaos)
        else:
            # "Writer's Block" panic - Scramble the sentence
            # 1. Shuffle order
            random.shuffle(candidate)
            # 2. Replace multiple words
            num_replace = int(len(candidate) * self.chaos_level)
            for _ in range(num_replace):
                idx = random.randint(0, len(candidate)-1)
                candidate[idx] = random.choice(VOCABULARY)
                
            return candidate, f"CHAOS ({self.chaos_level:.2f})"

    def run_cycle(self):
        # 1. Sense (Embed current sentence)
        curr_text = self.get_current_text()
        curr_vec = ENCODER.encode(curr_text)
        
        # 2. Think
        improved, dist = self.think(curr_vec)
        
        if improved:
            print(f"   >>> NEW BEST ({dist:.4f}): \"{curr_text}\"")
            
        # 3. Act
        new_sentence, mode = self.act()
        
        # 4. Simulate (Check if new sentence is better?)
        # In a real creative loop, we might just mutate and see.
        # Here, we employ the "Expectimax" lesson: Check before committing?
        # NO. We stick to Cybernetic Flow. We commit, then evaluate next cycle.
        # But to prevent "drifting away," we revert to BEST if we panic too long.
        
        if self.chaos_level > 0.6 and random.random() < 0.2:
            # "Reset to Draft" - Go back to best known state to try a different path
            self.sentence = list(self.best_sentence)
        else:
            self.sentence = new_sentence
            
        return dist, self.chaos_level

def main():
    print("--- BBIT SEMANTIC SCULPTOR ---")
    targets = [
        "A scary computer virus",
        "A peaceful morning in nature",
        "The heat of battle"
    ]
    
    target = random.choice(targets)
    print(f"Target Concept: '{target}'")
    print("-" * 40)
    
    agent = SemanticAgent(target, sentence_length=4)
    
    try:
        for t in range(500):
            dist, chaos = agent.run_cycle()
            
            if dist < 0.25: # Semantic similarity threshold
                print(f"\n>> CONVERGENCE ACHIEVED @ Step {t}")
                break
                
            time.sleep(0.01) # fast loop
            
    except KeyboardInterrupt:
        pass
        
    print("-" * 40)
    print(f"Target: '{target}'")
    print(f"Final Construction: \"{' '.join(agent.best_sentence)}\"")
    print(f"Semantic Distance: {agent.best_score:.4f}")

if __name__ == "__main__":
    main()