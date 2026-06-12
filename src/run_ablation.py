import pandas as pd
import numpy as np
import time
import os
import random
import csv
import matplotlib.pyplot as plt

try:
    from river import metrics, drift
    from river.ensemble import AdaptiveRandomForestClassifier
except ImportError:
    exit(1)

class StaticCSARF:
    def __init__(self, beta=300):
        self.beta = beta
        self.model = AdaptiveRandomForestClassifier(n_models=10, seed=42)
    def learn_one(self, x, y):
        k = np.random.poisson(self.beta if y == 1 else 1)
        for _ in range(k):
            self.model.learn_one(x, y)
        return self
    def predict_one(self, x): return self.model.predict_one(x)

class DynamicCSARF:
    def __init__(self, gamma=2.0, alpha=0.999, theta=0.99):
        self.gamma = gamma
        self.alpha = alpha
        self.theta = theta
        self.model = AdaptiveRandomForestClassifier(n_models=10, seed=42)
        
        self.count_maj = 0.0
        self.count_min = 0.0
        self.adwin = drift.ADWIN()
        self.d_t = 0.0
        
    def learn_one(self, x, y):
        self.count_maj = self.alpha * self.count_maj + (1 if y == 0 else 0)
        self.count_min = self.alpha * self.count_min + (1 if y == 1 else 0)
        ir_t = self.count_maj / max(1e-5, self.count_min)
        
        y_pred = self.model.predict_one(x)
        error = 0.0 if y_pred == y else 1.0
        self.adwin.update(error)
        
        if self.adwin.change_detected:
            self.d_t = 1.0
        else:
            self.d_t = self.theta * self.d_t
            
        if y == 1:
            lambda_t = min(1000.0, ir_t * (1.0 + self.gamma * self.d_t))
            k = np.random.poisson(max(1.0, lambda_t))
        else:
            k = np.random.poisson(1.0)
            
        for _ in range(k):
            self.model.learn_one(x, y)
        return self
        
    def predict_one(self, x): return self.model.predict_one(x)

def run_ablation():
    input_csv = r"data\ULB Credit Card Fraud Detection\creditcard.csv"
    print("Loading ULB dataset...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"File not found: {input_csv}. Please place the dataset in the correct location.")
        return
        
    frauds = df[df["Class"] == 1]
    legits = df[df["Class"] == 0]
    
    n_frauds = int(50000 * (len(frauds) / len(df)))
    n_legits = 50000 - n_frauds
    
    frauds_sample = frauds.sample(n=min(len(frauds), n_frauds), random_state=42)
    legits_sample = legits.sample(n=n_legits, random_state=42)
    
    df_subset = pd.concat([frauds_sample, legits_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    numeric_cols = df_subset.select_dtypes(include=['number']).columns
    df_subset = df_subset[numeric_cols]
    
    models_to_test = {
        r"Static ($\beta=100$)": StaticCSARF(beta=100),
        r"Static ($\beta=300$)": StaticCSARF(beta=300),
        "Dynamic (Proposed)": DynamicCSARF(gamma=2.0)
    }
    
    results_gmean = []
    results_recall = []
    labels = list(models_to_test.keys())
    
    for name, model in models_to_test.items():
        print(f"Testing {name}...")
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
        print(f"{name}: G-Mean = {metric_gmean.get():.4f}, Recall = {metric_recall.get():.4f}")
        
    # Plotting
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, results_gmean, width, label='G-Mean')
    rects2 = ax.bar(x + width/2, results_recall, width, label='Recall')

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Static vs Dynamic Penalty')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/ablation_study.pdf')
    print("Saved figures/ablation_study.pdf")

if __name__ == "__main__":
    run_ablation()

