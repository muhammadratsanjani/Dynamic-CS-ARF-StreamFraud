import pandas as pd
import numpy as np
import time
import os
import sys
import psutil
from river.ensemble import AdaptiveRandomForestClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.datasets import synth
from river import drift
import gc

sys.path.insert(0, os.path.dirname(__file__))
from run_benchmark import CSARFv2

random_seed = 42
np.random.seed(random_seed)

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
                    if np.random.rand() < self.minority_prob:
                        count += 1
                        yield x, y
                else:
                    count += 1
                    yield x, y
            except StopIteration:
                break

def get_generator(d_name):
    if d_name == "Agrawal":
        return ImbalancedStream(synth.Agrawal(seed=42), minority_prob=0.01, max_samples=50000)
    elif d_name == "Hyperplane":
        return ImbalancedStream(synth.Hyperplane(seed=42, mag_change=0.001, n_drift_features=2), minority_prob=0.01, max_samples=50000)
    else:
        raise ValueError("Unknown dataset")

def profile_model(d_name, m_name, model_class, kwargs):
    print(f"Profiling {m_name} on {d_name}...")
    gc.collect()
    
    process = psutil.Process(os.getpid())
    process.cpu_percent() # initial call
    mem_before = process.memory_info().rss
    
    model = model_class(**kwargs)
    gen = get_generator(d_name)
    
    start_time = time.time()
    n_transactions = 0
    
    for x, y in gen:
        y = int(y)
        model.predict_one(x)
        model.learn_one(x, y)
        n_transactions += 1
        
    end_time = time.time()
    exec_time = end_time - start_time
    
    cpu_usage = process.cpu_percent()
    
    import pickle
    model_size_bytes = len(pickle.dumps(model))
    model_size_mb = model_size_bytes / (1024 * 1024)
        
    tps = n_transactions / exec_time if exec_time > 0 else 0
    
    print(f"[{m_name}] TPS: {tps:.2f}, Size: {model_size_mb:.2f} MB, CPU: {cpu_usage:.2f}%, Time: {exec_time:.2f}s")
    
    return {
        "Dataset": d_name,
        "Model": m_name,
        "Throughput (TPS)": round(tps, 2),
        "Size (MB)": round(model_size_mb, 2),
        "CPU (%)": round(cpu_usage, 2),
        "Time (s)": round(exec_time, 2)
    }

def main():
    datasets = ["Agrawal", "Hyperplane"]
    models = [
        ("ARF (Standard)", AdaptiveRandomForestClassifier, {"n_models": 10, "seed": 42}),
        ("Dynamic CS-ARF (Proposed)", CSARFv2, {"gamma": 2.0, "alpha": 0.999, "lambda_cap": 100.0, "precision_threshold": 0.3})
    ]
    
    results = []
    for d_name in datasets:
        for m_name, m_class, m_kwargs in models:
            res = profile_model(d_name, m_name, m_class, m_kwargs)
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # Save the results
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "throughput_profiling.csv")
    df.to_csv(out_path, index=False)
    print(f"\nProfiling completed. Results saved to {out_path}")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
