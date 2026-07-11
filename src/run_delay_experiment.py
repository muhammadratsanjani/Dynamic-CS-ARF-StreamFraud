import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from collections import deque
from river import metrics
from river.ensemble import AdaptiveRandomForestClassifier

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom wrappers from run_benchmark
from run_benchmark import CSARFv2, stream_csv

def run_delayed_evaluation(model_class, kwargs, filepath, target_col, delay, max_rows=50000):
    model = model_class(**kwargs)
    metric_gmean = metrics.GeometricMean()
    metric_recall = metrics.Recall()
    
    # Latency Queue
    latency_queue = deque()
    
    start_time = time.time()
    count = 0
    
    # stream_csv yields (x, y)
    gen = stream_csv(filepath, target_col, max_rows=max_rows)
    
    for x, y in gen:
        # 1. Predict (Always immediate)
        y_pred = model.predict_one(x)
        if y_pred is not None:
            metric_gmean.update(y, y_pred)
            metric_recall.update(y, y_pred)
            
        # 2. Enqueue the true label for delayed learning
        latency_queue.append((x, y))
        
        # 3. Learn from the delayed transaction if queue is full
        if len(latency_queue) > delay:
            delayed_x, delayed_y = latency_queue.popleft()
            model.learn_one(delayed_x, delayed_y)
            
        count += 1
        
    end_time = time.time()
    print(f"Delay {delay}: G-Mean = {metric_gmean.get():.4f}, Recall = {metric_recall.get():.4f} (Time: {end_time-start_time:.2f}s)")
    
    return metric_gmean.get(), metric_recall.get()

def main():
    filepath = r"data\ULB Credit Card Fraud Detection\creditcard.csv"
    target_col = "Class"
    max_rows = 50000  # Subset for fast evaluation
    
    delays = [0, 100, 1000, 5000]
    
    results = {
        "Delay": [],
        "Standard ARF (G-Mean)": [],
        "Standard ARF (Recall)": [],
        "Dynamic CS-ARF (G-Mean)": [],
        "Dynamic CS-ARF (Recall)": []
    }
    
    print("Starting Delayed-Label Robustness Experiment...")
    
    for d in delays:
        print(f"\n--- Testing Delay = {d} transactions ---")
        
        # Standard ARF
        print("Evaluating Standard ARF...")
        arf_gmean, arf_recall = run_delayed_evaluation(
            AdaptiveRandomForestClassifier, {"n_models": 10, "seed": 42}, 
            filepath, target_col, d, max_rows
        )
        
        # Dynamic CS-ARF
        print("Evaluating Dynamic CS-ARF...")
        csarf_gmean, csarf_recall = run_delayed_evaluation(
            CSARFv2, {"gamma": 2.0, "alpha": 0.999, "lambda_cap": 100.0, "precision_threshold": 0.3},
            filepath, target_col, d, max_rows
        )
        
        results["Delay"].append(d)
        results["Standard ARF (G-Mean)"].append(arf_gmean)
        results["Standard ARF (Recall)"].append(arf_recall)
        results["Dynamic CS-ARF (G-Mean)"].append(csarf_gmean)
        results["Dynamic CS-ARF (Recall)"].append(csarf_recall)
        
    df = pd.DataFrame(results)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/delay_degradation.csv", index=False)
    
    # Plotting
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    plt.plot(df["Delay"], df["Dynamic CS-ARF (G-Mean)"], marker='o', linestyle='-', label="Dynamic CS-ARF (G-Mean)", color='blue', linewidth=2)
    plt.plot(df["Delay"], df["Standard ARF (G-Mean)"], marker='s', linestyle='--', label="Standard ARF (G-Mean)", color='red', linewidth=2)
    
    plt.xlabel("Verification Latency (Number of Transactions Delayed)")
    plt.ylabel("Geometric Mean (G-Mean)")
    plt.title("Performance Degradation under Delayed Labels (ULB Dataset)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("figures/delay_degradation.pdf")
    print("\nExperiment complete. Saved delay_degradation.csv and figures/delay_degradation.pdf.")

if __name__ == "__main__":
    main()
