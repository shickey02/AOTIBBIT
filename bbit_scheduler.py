#!/usr/bin/env python3
# BBIT TASK SCHEDULER (Load Balancer)
# Assigns 50 random jobs to 4 workers to minimize total time.
# Uses Apex Logic: Precision Swaps vs. Chaos Scrambles.

import random, time, copy
import numpy as np

# --- 1. THE ENVIRONMENT ---
class JobQueue:
    def __init__(self, num_jobs=50, num_workers=4):
        # Generate jobs with random durations (1 to 100 minutes)
        self.jobs = [random.randint(1, 100) for _ in range(num_jobs)]
        self.num_workers = num_workers
        
    def evaluate_schedule(self, schedule):
        """
        Input: List of worker assignments [0, 1, 0, 3, 2...] matching jobs.
        Output: The 'Makespan' (Time the last worker finishes).
        """
        worker_times = [0] * self.num_workers
        for job_idx, worker_idx in enumerate(schedule):
            worker_times[worker_idx] += self.jobs[job_idx]
            
        # The cost is the time of the busiest worker (The Bottleneck)
        makespan = max(worker_times)
        
        # Secondary objective: Balance. 
        # If two schedules have same makespan, prefer the one with less variance.
        imbalance = max(worker_times) - min(worker_times)
        
        return makespan, imbalance, worker_times

# --- 2. THE APEX SCHEDULER ---
class CyberneticScheduler:
    def __init__(self, num_jobs, num_workers):
        # State: Which worker gets which job?
        self.schedule = [random.randint(0, num_workers-1) for _ in range(num_jobs)]
        self.num_workers = num_workers
        
        self.best_schedule = list(self.schedule)
        self.best_score = float('inf')
        
        # Cybernetics
        self.stagnation = 0
        self.patience = 200
        self.chaos_level = 0.0
        self.cooling_down = 0 # Refractory period counter

    def think(self, current_score):
        # 1. Check for New Record
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_schedule = list(self.schedule)
            self.stagnation = 0
            self.chaos_level = 0.0
            self.cooling_down = 50 # Reward success with stability
            return True # Improved
        
        # 2. Cooling Phase (Forced Stability after Panic)
        if self.cooling_down > 0:
            self.cooling_down -= 1
            self.chaos_level = 0.0
            return False

        # 3. Anxiety Accumulation
        self.stagnation += 1
        
        if self.stagnation > self.patience:
            # RAMP UP CHAOS
            self.chaos_level = min(0.5, (self.stagnation - self.patience) / 500.0)
        else:
            self.chaos_level = 0.0
            
        return False

    def act(self):
        new_sched = list(self.schedule)
        
        # MODE A: PRECISION (Move one job)
        if self.chaos_level < 0.1:
            # Pick a random job and move it to a random worker
            # (In a smarter version, we'd move from Busiest to Laziest)
            job_idx = random.randint(0, len(new_sched)-1)
            new_worker = random.randint(0, self.num_workers-1)
            new_sched[job_idx] = new_worker
            return new_sched, "Precision"

        # MODE B: CHAOS (Reassign a chunk of jobs)
        else:
            # Scramble 10-50% of assignments
            num_changes = int(len(new_sched) * self.chaos_level)
            for _ in range(num_changes):
                idx = random.randint(0, len(new_sched)-1)
                new_sched[idx] = random.randint(0, self.num_workers-1)
            
            # Reset cooling to prevent spiral
            self.cooling_down = 20 
            return new_sched, f"CHAOS ({self.chaos_level:.2f})"

    def run_cycle(self, env):
        # Evaluate Current
        curr_max, curr_imb, _ = env.evaluate_schedule(self.schedule)
        score = curr_max + (curr_imb * 0.1) # Weighted score
        
        # Think
        improved = self.think(score)
        if improved:
            print(f"   >>> NEW RECORD: Max Time {curr_max} (Imbalance {curr_imb})")
        
        # Act
        candidate, mode = self.act()
        
        # Evaluate Candidate
        cand_max, cand_imb, _ = env.evaluate_schedule(candidate)
        cand_score = cand_max + (cand_imb * 0.1)
        
        gap = cand_score - score
        
        # Decision (Allow bad moves if Panicking)
        if gap < 0:
            self.schedule = candidate
        elif self.chaos_level > 0:
            # Acceptance probability
            if random.random() < self.chaos_level * 0.2:
                self.schedule = candidate # Accept bad move to escape
                
        return curr_max, self.chaos_level

def main():
    print("--- BBIT LOGISTICS SCHEDULER ---")
    env = JobQueue(num_jobs=50, num_workers=4)
    agent = CyberneticScheduler(50, 4)
    
    # Calculate Total Work
    total_work = sum(env.jobs)
    perfect_balance = total_work / 4
    print(f"Total Work: {total_work} mins")
    print(f"Theoretical Perfect Time: {perfect_balance:.1f} mins")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        for t in range(5000):
            makespan, chaos = agent.run_cycle(env)
            
            if t % 100 == 0:
                bar = "#" * int(chaos * 20)
                print(f"Step {t:04d} | Max Time: {makespan} | Chaos: [{bar:<10}]")
                
            # Convergence Check
            if makespan <= perfect_balance + 1: # Within 1 minute of perfection
                print(f"\n>> PERFECT BALANCE ACHIEVED @ Step {t}")
                break
                
    except KeyboardInterrupt:
        pass
        
    print("-" * 40)
    best_max, best_imb, loads = env.evaluate_schedule(agent.best_schedule)
    print(f"Final Solution Found in {time.time()-start_time:.2f}s")
    print(f"Worker Loads: {loads}")
    print(f"Bottleneck: {best_max} mins")
    print(f"Imbalance: {best_imb} mins")

if __name__ == "__main__":
    main()