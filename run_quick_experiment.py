import ERL
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing

# Configuration for Quick Verification
STRATEGIES = ['ERL', 'E', 'L', 'F', 'B']
TRIALS_PER_STRATEGY = 5 # Small number for quick check
MAX_STEPS = 2000 # Reduced from 1,000,000

def run_single_trial(strategy, trial_num):
    # Use trial_num as seed for reproducibility across strategies
    # print(f"[{strategy}] Starting Trial {trial_num+1}/{TRIALS_PER_STRATEGY}")
    steps = ERL.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=trial_num)
    return steps

def run_experiments():
    results = {s: [] for s in STRATEGIES}
    
    print(f"Running {TRIALS_PER_STRATEGY} trials (max {MAX_STEPS} steps) for each strategy using {multiprocessing.cpu_count()} CPU cores...")
    
    # Submit ALL tasks at once
    with multiprocessing.Pool() as pool:
        all_async_results = []
        
        for strategy in STRATEGIES:
            for i in range(TRIALS_PER_STRATEGY):
                res = pool.apply_async(run_single_trial, (strategy, i))
                all_async_results.append((strategy, i, res))
        
        start_time = time.time()
        total_trials = len(all_async_results)
        finished_count = 0
        
        while finished_count < total_trials:
            for idx, (strategy, trial_num, res) in enumerate(all_async_results):
                if res is None: continue
                
                if res.ready():
                    try:
                        steps = res.get()
                        results[strategy].append(steps)
                        all_async_results[idx] = (strategy, trial_num, None)
                        finished_count += 1
                        # print(f"[{strategy}] Trial {trial_num+1} finished: {steps}")
                    except Exception as e:
                        print(f"[{strategy}] Trial {trial_num+1} failed: {e}")
                        all_async_results[idx] = (strategy, trial_num, None)
                        finished_count += 1
            
            time.sleep(0.5)
            
        duration = time.time() - start_time
        print(f"All experiments finished in {duration:.2f}s")
    
    # Analyze Results
    print("\n--- RESULTS (Average Steps Survived) ---")
    for s in STRATEGIES:
        avg = np.mean(results[s]) if results[s] else 0
        med = np.median(results[s]) if results[s] else 0
        survived_full = sum(1 for x in results[s] if x >= MAX_STEPS)
        print(f"{s}: Avg={avg:.1f}, Median={med:.1f}, Survived Full Duration={survived_full}/{TRIALS_PER_STRATEGY}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    run_experiments()
