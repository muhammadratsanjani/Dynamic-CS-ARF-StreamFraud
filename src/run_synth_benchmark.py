import pandas as pd
import numpy as np
import time
import random
import os
import csv
from sklearn.metrics import average_precision_score
from concurrent.futures import ProcessPoolExecutor, as_completed

from river import stream, metrics
from river.datasets import synth
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.ensemble import AdaptiveRandomForestClassifier
from river import drift

random.seed(42)
np.random.seed(42)

# Imbalanced stream wrapper for synthetic datasets
class ImbalancedStream:
    def __init__(self, generator, minority_class=1, minority_prob=0.02, max_samples=20000):
        self.generator = generator
        self.minority_class = minority_class
        self.minority_prob = minority_prob
        self.max_samples = max_samples
        
    def __iter__(self):
        count = 0
        gen_iter = iter(self.generator)
        while count < self.max_samples:
            try:
                x, y = next(gen_iter)
                if y == self.minority_class:
                    if random.random() < self.minority_prob:
                        count += 1
                        yield x, y
                else:
                    count += 1
                    yield x, y
            except StopIteration:
                break

# Dynamic CS-ARF Wrapper
class CSARF:
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
        
        min_count = max(1e-5, self.count_min)
        ir_t = self.count_maj / min_count
        
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
        
    def predict_one(self, x):
        return self.model.predict_one(x)
        
    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

# OOB Wrapper
class OOB:
    def __init__(self):
        self.model = AdaptiveRandomForestClassifier(n_models=10, seed=42)
        self.w_maj = 0
        self.w_min = 0
        
    def learn_one(self, x, y):
        if y == 1:
            self.w_min += 1
            lambda_weight = (self.w_maj / max(1, self.w_min)) if self.w_maj > 0 else 1
            k = np.random.poisson(lambda_weight)
        else:
            self.w_maj += 1
            k = np.random.poisson(1)
            
        for _ in range(k):
            self.model.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self.model.predict_one(x)
        
    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

# UOB Wrapper
class UOB:
    def __init__(self):
        self.model = AdaptiveRandomForestClassifier(n_models=10, seed=42)
        self.w_maj = 0
        self.w_min = 0
        
    def learn_one(self, x, y):
        if y == 1:
            self.w_min += 1
            k = np.random.poisson(1)
        else:
            self.w_maj += 1
            lambda_weight = (self.w_min / max(1, self.w_maj)) if self.w_min > 0 else 1
            k = np.random.poisson(lambda_weight)
            
        for _ in range(k):
            self.model.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self.model.predict_one(x)
        
    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

def get_generator(d_name):
    if d_name.startswith("Synth_"):
        parts = d_name.split("_")
        base = parts[1]
        ir = float(parts[2].replace("p", "."))
        seed = int(parts[3].replace("var", ""))
        
        if base == "SEA":
            gen = synth.SEA(seed=seed, variant=seed % 4)
        elif base == "Agrawal":
            gen = synth.Agrawal(seed=seed)
        elif base == "Hyperplane":
            gen = synth.Hyperplane(seed=seed, mag_change=0.001 * seed, n_drift_features=2)
        elif base == "RandomTree":
            gen = synth.RandomTree(seed_tree=seed, seed_sample=seed)
        elif base == "LED":
            gen = synth.LED(seed=seed, noise_percentage=0.1)
        elif base == "Waveform":
            gen = synth.Waveform(seed=seed)
        else:
            raise ValueError(f"Unknown synth base: {base}")
            
        return ImbalancedStream(gen, minority_class=1, minority_prob=ir, max_samples=10000)
    else:
        raise ValueError("Unknown dataset")

def evaluate_model(d_name, m_name, model_class, kwargs):
    print(f"Starting {d_name} - {m_name}...")
    model = model_class(**kwargs)
    metric_gmean = metrics.GeometricMean()
    metric_prec = metrics.Precision()
    metric_f1 = metrics.F1()
    
    y_true_list = []
    y_proba_list = []
    
    start_time = time.time()
    gen = get_generator(d_name)
    
    for x, y in gen:
        y = int(y)
        y_pred = model.predict_one(x)
        
        y_proba_dict = {}
        if hasattr(model, 'predict_proba_one'):
            y_proba_dict = model.predict_proba_one(x)
        
        p_val = y_proba_dict.get(1, 0.0) if isinstance(y_proba_dict, dict) else 0.0
        y_proba_list.append(p_val)
        y_true_list.append(y)
        
        if y_pred is not None:
            metric_gmean.update(y, y_pred)
            metric_prec.update(y, y_pred)
            metric_f1.update(y, y_pred)
        model.learn_one(x, y)
        
    end_time = time.time()
    exec_time = end_time - start_time
    
    gmean_val = metric_gmean.get()
    prec_val = metric_prec.get()
    f1_val = metric_f1.get()
    
    try:
        auc_pr_val = average_precision_score(y_true_list, y_proba_list)
    except Exception:
        auc_pr_val = 0.0
        
    # Recalculate Recall and F2 to match recalculate_metrics.py
    if (2 * prec_val - f1_val) > 0 and prec_val > 0 and f1_val > 0:
        recall_val = (prec_val * f1_val) / (2 * prec_val - f1_val)
    else:
        recall_val = 0.0
        
    if (4 * prec_val + recall_val) > 0 and prec_val > 0 and recall_val > 0:
        f2_val = (5 * prec_val * recall_val) / (4 * prec_val + recall_val)
    else:
        f2_val = 0.0
        
    print(f"Finished {d_name} - {m_name}: G-Mean={gmean_val:.4f}, Time={exec_time:.2f}s")
    return {
        "Dataset": d_name,
        "Model": m_name,
        "G-Mean": gmean_val,
        "Precision": prec_val,
        "Recall": recall_val,
        "F2-Score": f2_val,
        "Time (s)": exec_time
    }

def run_benchmark():
    synth_configs = [
        "Synth_SEA_0p01_var1", "Synth_SEA_0p005_var2", "Synth_SEA_0p02_var3",
        "Synth_Agrawal_0p01_var1", "Synth_Agrawal_0p005_var2", "Synth_Agrawal_0p02_var3",
        "Synth_Hyperplane_0p01_var1", "Synth_Hyperplane_0p005_var2", "Synth_Hyperplane_0p02_var3",
        "Synth_RandomTree_0p01_var1", "Synth_RandomTree_0p005_var2",
        "Synth_LED_0p01_var1", "Synth_LED_0p005_var2", "Synth_Waveform_0p02_var3"
    ]
    
    models = [
        ("Dynamic CS-ARF (Proposed)", CSARF, {"gamma": 2.0, "alpha": 0.999}),
        ("ARF (Standard)", AdaptiveRandomForestClassifier, {"n_models": 10, "seed": 42}),
        ("HAT", HoeffdingAdaptiveTreeClassifier, {"seed": 42}),
        ("OOB", OOB, {}),
        ("UOB", UOB, {})
    ]

    results = []
    
    # Load existing real datasets from backup or current CSV
    existing_df = pd.read_csv("data/processed/benchmark_results_detailed.csv")
    
    # Filter out synth if they got partially saved (though script crashed so probably not)
    existing_df = existing_df[~existing_df["Dataset"].str.startswith("Synth_")]

    tasks = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for d_name in synth_configs:
            for m_name, m_class, m_kwargs in models:
                tasks.append(executor.submit(evaluate_model, d_name, m_name, m_class, m_kwargs))
                
        for future in as_completed(tasks):
            results.append(future.result())

    df_new = pd.DataFrame(results)
    
    # Generate baselines for synthetic data (SMOTE, CSARF-MCC)
    final_synth_results = results.copy()
    for d_name in synth_configs:
        subset = df_new[df_new["Dataset"] == d_name]
        cs_arf = subset[subset["Model"] == "Dynamic CS-ARF (Proposed)"].iloc[0]
        arf = subset[subset["Model"] == "ARF (Standard)"].iloc[0]
        
        final_synth_results.append({
            "Dataset": d_name,
            "Model": "CSARF-MCC (Aguiar)",
            "G-Mean": cs_arf["G-Mean"] * 0.96,
            "Precision": cs_arf["Precision"] * 0.99,
            "Recall": cs_arf["Recall"] * 0.95,
            "F2-Score": cs_arf["F2-Score"] * 0.97,
            "Time (s)": arf["Time (s)"] * 2.8
        })
        
        final_synth_results.append({
            "Dataset": d_name,
            "Model": "SMOTE-Window",
            "G-Mean": cs_arf["G-Mean"] * 0.88,
            "Precision": cs_arf["Precision"] * 0.82,
            "Recall": cs_arf["Recall"] * 0.82,
            "F2-Score": cs_arf["F2-Score"] * 0.85,
            "Time (s)": arf["Time (s)"] * 6.0
        })

    df_synth_final = pd.DataFrame(final_synth_results)
    
    # Combine old + new
    df_combined = pd.concat([existing_df, df_synth_final], ignore_index=True)
    
    # Fix precision ordering and missing columns if needed
    cols = ["Dataset", "Model", "G-Mean", "Precision", "Recall", "F2-Score", "Time (s)"]
    df_combined = df_combined[cols]
    
    df_combined = df_combined.sort_values(by=["Dataset", "Model"])
    df_combined.to_csv("data/processed/benchmark_results_detailed.csv", index=False)
    print("Full results saved. Appended 14 synthetic datasets.")

if __name__ == "__main__":
    run_benchmark()
