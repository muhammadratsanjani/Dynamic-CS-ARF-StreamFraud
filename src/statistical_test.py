import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def run_stats():
    df = pd.read_csv("data/processed/benchmark_results_detailed.csv")
    # Pivot to Rows=Dataset, Cols=Model
    pivot = df.pivot(index="Dataset", columns="Model", values="F2-Score")
    
    print("Matrix of G-Mean:")
    print(pivot)
    
    # Ranks (descending: highest G-mean = rank 1)
    ranks = pivot.rank(axis=1, ascending=False)
    
    avg_ranks = ranks.mean()
    print("\nAverage Ranks:")
    print(avg_ranks.sort_values())
    
    # Friedman test
    stat, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    print(f"\nFriedman Test: Statistic={stat:.4f}, p-value={p:.4e}")
    
    if p < 0.05:
        # Nemenyi
        # Nemenyi requires data formatted as a 2D array [samples x groups] but scikit-posthocs might want flat arrays
        # Wait, posthoc_nemenyi_friedman takes a 2D array block.
        nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
        nemenyi.columns = pivot.columns
        nemenyi.index = pivot.columns
        print("\nNemenyi p-values:")
        print(nemenyi)
        
        # Calculate CD roughly (for alpha=0.05, N=5 datasets, k=7 algorithms)
        # q_alpha = 2.949 (from Nemenyi tables for k=7, alpha=0.05)
        k = len(pivot.columns)
        n = len(pivot)
        cd = 2.949 * np.sqrt(k * (k + 1) / (6 * n))
        print(f"\nCritical Difference (CD) = {cd:.4f}")
    else:
        print("Friedman test not significant, skipping post-hoc.")

if __name__ == "__main__":
    try:
        run_stats()
    except Exception as e:
        print(f"Error in stats: {e}")
