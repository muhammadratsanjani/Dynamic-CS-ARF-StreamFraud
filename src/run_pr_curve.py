import pandas as pd
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from river.ensemble import AdaptiveRandomForestClassifier
from river import drift

random_seed = 42
np.random.seed(random_seed)

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

def stream_csv(filepath, target_col, max_rows=50000):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= max_rows:
                break
            y_str = row.pop(target_col, "0")
            if not y_str: y_str = "0"
            y = int(float(y_str))
            
            x = {}
            for k, v in row.items():
                if v is None or v == '':
                    continue
                try:
                    x[k] = float(v)
                except ValueError:
                    pass
            yield x, y
            count += 1

def get_generator(d_name):
    # Fixed to 50000 for fast execution consistent with prior subsets
    if d_name == "ULB":
        return stream_csv(r"data\ULB Credit Card Fraud Detection\creditcard.csv", "Class", max_rows=50000)
    elif d_name == "PaySim":
        return stream_csv(r"data\PaySim Synthetic Mobile Money Fraud\PS_20174392719_1491204439457_log.csv", "isFraud", max_rows=50000)
    elif d_name == "IEEE-CIS":
        return stream_csv(r"data\IEEE-CIS\train_transaction.csv", "isFraud", max_rows=50000)
    else:
        raise ValueError("Unknown dataset")

def collect_predictions(d_name, m_name, model_class, kwargs):
    print(f"Running {m_name} on {d_name}...")
    model = model_class(**kwargs)
    gen = get_generator(d_name)
    
    y_true_list = []
    y_proba_list = []
    
    for x, y in gen:
        y = int(y)
        
        y_proba_dict = {}
        if hasattr(model, 'predict_proba_one'):
            y_proba_dict = model.predict_proba_one(x)
            
        p_val = y_proba_dict.get(1, 0.0) if isinstance(y_proba_dict, dict) else 0.0
        
        y_proba_list.append(p_val)
        y_true_list.append(y)
        
        model.learn_one(x, y)
        
    auc_pr = average_precision_score(y_true_list, y_proba_list)
    print(f"[{m_name} on {d_name}] PR-AUC: {auc_pr:.4f}")
    
    return y_true_list, y_proba_list, auc_pr

def main():
    datasets = ["ULB", "PaySim", "IEEE-CIS"]
    models = [
        ("ARF", AdaptiveRandomForestClassifier, {"n_models": 10, "seed": 42}),
        ("Dynamic CS-ARF", CSARF, {"gamma": 2.0, "alpha": 0.999})
    ]
    
    # Matplotlib setup
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    results = []

    for i, d_name in enumerate(datasets):
        ax = axes[i]
        
        for m_name, m_class, m_kwargs in models:
            y_true, y_proba, auc_pr = collect_predictions(d_name, m_name, m_class, m_kwargs)
            
            # Save results for text table
            results.append({
                "Dataset": d_name,
                "Model": m_name,
                "PR-AUC": round(auc_pr, 4)
            })
            
            # Compute PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            
            # Plot
            line_style = '-' if 'CS-ARF' in m_name else '--'
            color = 'blue' if 'CS-ARF' in m_name else 'red'
            ax.plot(recall, precision, linestyle=line_style, color=color, 
                    label=f'{m_name} (AUC = {auc_pr:.3f})')
            
        ax.set_title(f'{d_name} Dataset')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pr_curves.pdf")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to {out_path}")
    
    df = pd.DataFrame(results)
    df.to_csv("data/processed/pr_auc_results.csv", index=False)
    print("\nPR-AUC Scores:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
