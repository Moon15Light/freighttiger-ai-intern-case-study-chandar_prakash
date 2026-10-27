"""
Reproducibility Verification Harness
Verifies that 3 consecutive runs produce 100% byte-identical numerical and verdict results.
"""

import pandas as pd
from solution import run_pipeline, OUTPUT_CSV_PATH

def verify_reproducibility():
    runs = []
    
    print("Executing Reproducibility Test (3 Consecutive Runs)...")
    for run_num in range(1, 4):
        print(f"--- Running Test Iteration {run_num} ---")
        run_pipeline()
        df = pd.read_csv(OUTPUT_CSV_PATH)
        runs.append(df)
        
    diff_1_2 = runs[0].equals(runs[1])
    diff_2_3 = runs[1].equals(runs[2])
    
    if diff_1_2 and diff_2_3:
        print("\nSUCCESS: 100% Reproducibility Verified Across 3 Runs! Zero diffs detected.")
    else:
        print("\nFAILURE: Mismatch detected between pipeline runs.")

if __name__ == "__main__":
    verify_reproducibility()
