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
        
def thin_minority(generator, minority_class=1, minority_prob=0.01, rng=None):
    """Yield from `generator` keeping all majority instances but only a
    `minority_prob` fraction of minority instances, to induce severe
    class imbalance while preserving the generator's natural concept."""
    rng = rng or random
    for x, y in generator:
        if y == minority_class:
            if rng.random() < minority_prob:
                yield x, y
        else:
            yield x, y

class ImbalancedDriftStream:
    """Switches between three thinned Agrawal concepts at exact OUTPUT
    transaction indices (t=20000 and t=40000), so the drift markers plotted
    on the x-axis line up with when the concept actually changes.

    NOTE: a previous version fed `position=20000/40000` to river's
    ConceptDriftStream, but that position is measured in RAW (pre-thinning)
    generator draws, not in the thinned OUTPUT stream that gets plotted.
    With ~1% minority thinning, the raw and output indices diverge by a
    factor of ~1.6-2x, so the drift was actually landing at output index
    ~12,400 and ~21,800 while the figure labeled it at 20,000 and 40,000.
    """
    def __init__(self, switch_points=(20000, 40000), minority_class=1,
                 minority_prob=0.01, max_samples=60000, seed=42):
        self.switch_points = switch_points
        self.minority_class = minority_class
        self.minority_prob = minority_prob
        self.max_samples = max_samples
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        concepts = [
            synth.Agrawal(classification_function=1, seed=self.seed),
            synth.Agrawal(classification_function=2, seed=self.seed),
            synth.Agrawal(classification_function=3, seed=self.seed),
        ]
        boundaries = list(self.switch_points) + [self.max_samples]
        count = 0
        for concept, upper in zip(concepts, boundaries):
            for x, y in thin_minority(concept, self.minority_class, self.minority_prob, rng):
                if count >= upper:
                    break
                count += 1
                yield x, y
            if count >= self.max_samples:
                break

def generate_drift_stream():
    # Base stream configurations (Agrawal is good because drift is very clear).
    # Concept switches at t=20,000 and t=40,000 are aligned to the OUTPUT
    # (post-thinning) transaction index, matching the plotted x-axis.
    return ImbalancedDriftStream(switch_points=(20000, 40000), minority_class=1,
                                  minority_prob=0.01, max_samples=60000, seed=42)

def track_model_performance(model_name, model_obj, stream_gen):
    print(f"Tracking {model_name}...")
    # window_size=1000 with a 1% minority rate yields ~10 positive instances
    # per window on average, which is too sparse for a stable Recall estimate
    # (Recall is undefined/0 whenever a window happens to contain 0 or few
    # fraud cases). A window of 3000 (~30 expected positives) is used instead
    # to reduce this small-sample noise while still tracking local recovery.
    rolling_recall = metrics.Rolling(metrics.Recall(), window_size=3000)
    
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
    plt.ylabel('Rolling Recall (Window=3000)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0, 60000])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_path = 'figures/drift_adaptation.pdf'
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")

    # Export the raw series and summary stats used in the manuscript narrative.
    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame({
        't_arf': pd.Series(t_arf), 'recall_arf': pd.Series(recall_arf),
        't_csarf': pd.Series(t_csarf), 'recall_csarf': pd.Series(recall_csarf),
    }).to_csv('data/processed/drift_rolling_recall.csv', index=False)

    def segment_max(t_list, r_list, lo, hi):
        vals = [r for t, r in zip(t_list, r_list) if lo <= t <= hi]
        return max(vals) if vals else float('nan')

    print("\n=== Segment-wise peak Rolling Recall ===")
    for lo, hi, name in [(0, 20000, 'pre-drift1'), (20000, 40000, 'between drift1-drift2'), (40000, 60000, 'post-drift2')]:
        print(f"{name:25s} ARF peak={segment_max(t_arf, recall_arf, lo, hi):.4f}  Dynamic peak={segment_max(t_csarf, recall_csarf, lo, hi):.4f}")
    print(f"{'overall':25s} ARF peak={max(recall_arf):.4f}  Dynamic peak={max(recall_csarf):.4f}")
    print(f"{'overall':25s} ARF mean={sum(recall_arf)/len(recall_arf):.4f}  Dynamic mean={sum(recall_csarf)/len(recall_csarf):.4f}")

if __name__ == "__main__":
    main()
