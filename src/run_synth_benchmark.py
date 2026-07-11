import pandas as pd
import numpy as np
import time
import random
import os
import csv
from collections import deque
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

# Cost-Sensitive ARF with MCC-weighted voting (Loezer et al. 2020; Aguiar et al.
# 2023) -- see src/run_benchmark.py for full documentation of the design.
class CSARFMCC:
    def __init__(self, n_models=10, seed=42):
        self.model = AdaptiveRandomForestClassifier(n_models=n_models, seed=seed)
        self.mcc = [metrics.MCC() for _ in range(n_models)]

    def _weighted_votes(self, x):
        votes, total_w = {}, 0.0
        for member, mcc in zip(self.model.models, self.mcc):
            pred = member.predict_one(x)
            if pred is None:
                continue
            w = max(mcc.get(), 0.05)
            votes[pred] = votes.get(pred, 0.0) + w
            total_w += w
        return votes, total_w

    def predict_one(self, x):
        votes, _ = self._weighted_votes(x)
        if not votes:
            return None
        return max(votes, key=votes.get)

    def predict_proba_one(self, x):
        votes, total_w = self._weighted_votes(x)
        if total_w == 0:
            return {0: 1.0, 1: 0.0}
        proba = {0: 0.0, 1: 0.0}
        proba.update({k: v / total_w for k, v in votes.items()})
        return proba

    def learn_one(self, x, y):
        for member, mcc in zip(self.model.models, self.mcc):
            pred = member.predict_one(x)
            if pred is not None:
                mcc.update(y, pred)
        self.model.learn_one(x, y)
        return self

# Windowed SMOTE for streams (C-SMOTE/SMOTE-OB style) -- see
# src/run_benchmark.py for full documentation of the design.
class SMOTEWindow:
    def __init__(self, n_models=10, window_size=200, k_synthetic=5, seed=42):
        self.model = AdaptiveRandomForestClassifier(n_models=n_models, seed=seed)
        self.window = deque(maxlen=window_size)
        self.k_synthetic = k_synthetic
        self.rng = random.Random(seed)

    def predict_one(self, x):
        return self.model.predict_one(x)

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def _interpolate(self, x, neighbor):
        synth_x = {}
        alpha = self.rng.random()
        for key, val in x.items():
            n_val = neighbor.get(key)
            if isinstance(val, (int, float)) and isinstance(n_val, (int, float)):
                synth_x[key] = val + alpha * (n_val - val)
            else:
                synth_x[key] = val
        return synth_x

    def learn_one(self, x, y):
        if y == 1:
            if len(self.window) >= 1:
                for _ in range(self.k_synthetic):
                    neighbor = self.rng.choice(self.window)
                    self.model.learn_one(self._interpolate(x, neighbor), 1)
            self.window.append(x)
        self.model.learn_one(x, y)
        return self

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
        ("UOB", UOB, {}),
        ("CSARF-MCC (Aguiar)", CSARFMCC, {"n_models": 10, "seed": 42}),
        ("SMOTE-Window", SMOTEWindow, {"n_models": 10, "seed": 42}),
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

    df_synth_final = pd.DataFrame(results)

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
