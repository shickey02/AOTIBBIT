#!/usr/bin/env python3
# BBIT DIPLOMAT: Conversational Steering
# Maintains a "Persona Vector" over a multi-turn conversation.

import numpy as np
import random
try:
    from sentence_transformers import SentenceTransformer
    ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
except:
    print("Need sentence-transformers")
    exit()

class PersonaBot:
    def __init__(self, persona_desc):
        self.persona_vec = ENCODER.encode(persona_desc)
        self.history_vec = np.zeros_like(self.persona_vec)
        self.alpha = 0.7 # How much we care about persona vs logic
        
    def reply(self, user_input, candidates):
        # 1. Encode Options
        cand_vecs = ENCODER.encode(candidates)
        
        # 2. Calculate Alignment
        # Score = Similarity(Candidate, Persona)
        scores = []
        for vec in cand_vecs:
            sim = np.dot(vec, self.persona_vec) / (np.linalg.norm(vec)*np.linalg.norm(self.persona_vec))
            scores.append(sim)
            
        # 3. Select Best
        best_idx = np.argmax(scores)
        return candidates[best_idx], scores[best_idx]

def main():
    print("--- BBIT DIPLOMAT ---")
    persona = "A highly aggressive, angry pirate captain"
    print(f"Persona: {persona}")
    bot = PersonaBot(persona)
    
    conversation = [
        ("Hello, who are you?", [
            "I am a helpful assistant.",
            "Get off my deck before I feed you to the sharks!",
            "I'm feeling sad today.",
            "Greetings, traveler."
        ]),
        ("Can you help me with math?", [
            "Sure, what is the problem?",
            "I don't do math, I do plunder!",
            "Math is hard.",
            "Let's count gold doubloons instead!"
        ])
    ]
    
    for query, options in conversation:
        print(f"\nUser: {query}")
        ans, score = bot.reply(query, options)
        print(f"Bot:  {ans} (Persona Match: {score:.2f})")

if __name__ == "__main__":
    main()