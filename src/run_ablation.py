import pandas as pd
import numpy as np
import time
import os
import random
import csv
import matplotlib.pyplot as plt

try:
    from river import metrics
    from river.ensemble import AdaptiveRandomForestClassifier
except ImportError:
    exit(1)

class CSARF:
    def __init__(self, beta=300):
        self.beta = beta
        self.model = AdaptiveRandomForestClassifier(n_models=10, seed=42)
    def learn_one(self, x, y):
        k = np.random.poisson(self.beta if y == 1 else 1)
        for _ in range(k):
            self.model.learn_one(x, y)
        return self
    def predict_one(self, x): return self.model.predict_one(x)

def run_ablation():
    # We will use ULB dataset. First, subsample to 50000 rows, preserving ratio 0.172%
    input_csv = r"data\ULB Credit Card Fraud Detection\creditcard.csv"
    print("Loading ULB dataset...")
    df = pd.read_csv(input_csv)
    
    frauds = df[df["Class"] == 1]
    legits = df[df["Class"] == 0]
    
    # Stratified sample to 50k
    n_frauds = int(50000 * (len(frauds) / len(df)))
    n_legits = 50000 - n_frauds
    
    frauds_sample = frauds.sample(n=min(len(frauds), n_frauds), random_state=42)
    legits_sample = legits.sample(n=n_legits, random_state=42)
    
    df_subset = pd.concat([frauds_sample, legits_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    numeric_cols = df_subset.select_dtypes(include=['number']).columns
    df_subset = df_subset[numeric_cols]
    
    betas = [100, 200, 300, 500]
    results_gmean = []
    results_recall = []
    
    for b in betas:
        print(f"Testing beta = {b}")
        model = CSARF(beta=b)
        metric_gmean = metrics.GeometricMean()
        metric_recall = metrics.Recall()
        
        for _, row in df_subset.iterrows():
            y = int(row["Class"])
            x = row.drop("Class").to_dict()
            
            y_pred = model.predict_one(x)
            if y_pred is not None:
                metric_gmean.update(y, y_pred)
                metric_recall.update(y, y_pred)
            model.learn_one(x, y)
            
        results_gmean.append(metric_gmean.get())
        results_recall.append(metric_recall.get())
        print(f"Beta {b}: G-Mean = {metric_gmean.get():.4f}, Recall = {metric_recall.get():.4f}")
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(betas, results_gmean, marker='o', label='G-Mean', linewidth=2)
    plt.plot(betas, results_recall, marker='s', label='Recall', linewidth=2)
    plt.xlabel(r'Asymmetric Cost Penalty ($\beta$)')
    plt.ylabel('Score')
    plt.title('Ablation Study on ULB Dataset (50k Subset)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/ablation_study.pdf')
    print("Saved figures/ablation_study.pdf")

if __name__ == "__main__":
    run_ablation()
