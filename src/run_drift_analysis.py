import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
from river import metrics
from river.datasets import synth
from river.ensemble import AdaptiveRandomForestClassifier
from river import drift

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

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
        
class ImbalancedStream:
    def __init__(self, generator, minority_class=1, minority_prob=0.01, max_samples=10000):
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

def generate_drift_stream():
    # Base stream configurations (Agrawal is good because drift is very clear)
    stream1 = synth.Agrawal(classification_function=1, seed=42)
    stream2 = synth.Agrawal(classification_function=2, seed=42)
    stream3 = synth.Agrawal(classification_function=3, seed=42)

    # Induce drift at 20k and 40k
    drift1 = synth.ConceptDriftStream(stream=stream1, drift_stream=stream2, position=20000, width=500, seed=42)
    drift2 = synth.ConceptDriftStream(stream=drift1, drift_stream=stream3, position=40000, width=500, seed=42)
    
    # Wrap in ImbalancedStream to make minority class 1% of the stream, total 60k
    return ImbalancedStream(drift2, minority_class=1, minority_prob=0.01, max_samples=60000)

def track_model_performance(model_name, model_obj, stream_gen):
    print(f"Tracking {model_name}...")
    rolling_recall = metrics.Rolling(metrics.Recall(), window_size=1000)
    
    t_list = []
    recall_list = []
    
    t = 0
    for x, y in stream_gen:
        t += 1
        y_pred = model_obj.predict_one(x)
        
        if y_pred is not None:
            rolling_recall.update(y, y_pred)
            
        model_obj.learn_one(x, y)
        
        # Log metric every 500 steps to keep the plot smooth but not overly dense
        if t % 500 == 0:
            t_list.append(t)
            recall_list.append(rolling_recall.get())
            
    return t_list, recall_list

def main():
    # Initialize models
    arf = AdaptiveRandomForestClassifier(n_models=10, seed=42)
    csarf = CSARF(gamma=2.0, alpha=0.999)
    
    # Run ARF
    gen_arf = generate_drift_stream()
    t_arf, recall_arf = track_model_performance("ARF (Standard)", arf, gen_arf)
    
    # Run CS-ARF
    gen_csarf = generate_drift_stream()
    t_csarf, recall_csarf = track_model_performance("Dynamic CS-ARF", csarf, gen_csarf)
    
    # Plotting
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 5))
    
    plt.plot(t_arf, recall_arf, label='ARF (Standard)', linestyle='--', color='red', alpha=0.8, linewidth=2)
    plt.plot(t_csarf, recall_csarf, label='Dynamic CS-ARF (Proposed)', linestyle='-', color='blue', linewidth=2)
    
    # Drift markers
    plt.axvline(x=20000, color='gray', linestyle=':', linewidth=2)
    plt.axvline(x=40000, color='gray', linestyle=':', linewidth=2)
    
    # Annotations
    plt.text(20500, 0.1, 'Concept Drift 1', rotation=90, color='gray', fontsize=12)
    plt.text(40500, 0.1, 'Concept Drift 2', rotation=90, color='gray', fontsize=12)
    
    plt.title('Rolling Recall vs Time (Agrawal Stream with 1% Imbalance)')
    plt.xlabel('Transactions Processed')
    plt.ylabel('Rolling Recall (Window=1000)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0, 60000])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_path = 'figures/drift_adaptation.pdf'
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    main()
