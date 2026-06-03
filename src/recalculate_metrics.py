import pandas as pd
import numpy as np

def recalculate():
    df = pd.read_csv("data/processed/benchmark_results_detailed.csv")
    
    recalls = []
    f2_scores = []
    
    for _, row in df.iterrows():
        P = row["Precision"]
        F1 = row["F1"]
        
        # Calculate Recall
        if (2 * P - F1) > 0 and P > 0 and F1 > 0:
            R = (P * F1) / (2 * P - F1)
        else:
            R = 0.0
            
        # Calculate F2
        if (4 * P + R) > 0 and P > 0 and R > 0:
            F2 = (5 * P * R) / (4 * P + R)
        else:
            F2 = 0.0
            
        recalls.append(R)
        f2_scores.append(F2)
        
    df["Recall"] = recalls
    df["F2-Score"] = f2_scores
    
    # We will keep G-Mean, Precision, Recall, F2-Score, Time
    # Drop F1 and AUC-PR to make space for the new metrics focus
    if "F1" in df.columns: df = df.drop(columns=["F1"])
    if "AUC-PR" in df.columns: df = df.drop(columns=["AUC-PR"])
    
    # Reorder columns
    cols = ["Dataset", "Model", "G-Mean", "Precision", "Recall", "F2-Score", "Time (s)"]
    df = df[cols]
    
    df.to_csv("data/processed/benchmark_results_detailed.csv", index=False)
    print("Recalculated metrics. Saved to CSV.")

if __name__ == "__main__":
    recalculate()
