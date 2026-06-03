import pandas as pd
import os

def create_mini_dataset(input_csv, output_csv, target_col, max_rows=30000, max_frauds=None):
    print(f"Processing {input_csv}...")
    try:
        # For huge files, chunking is better, but 6.3M fits in 16GB RAM
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error loading {input_csv}: {e}")
        return
        
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    
    frauds = df[df[target_col] == 1]
    legits = df[df[target_col] == 0]
    
    # Calculate original imbalance ratio
    total_orig = len(df)
    fraud_orig = len(frauds)
    ratio = fraud_orig / max(total_orig, 1)
    
    if max_frauds is not None:
        if len(frauds) > max_frauds:
            frauds = frauds.sample(n=max_frauds, random_state=42)
            
    # Based on number of frauds we keep, we pull enough legits to match original ratio
    # Or just cap total rows to max_rows
    target_legits_count = int(len(frauds) / max(ratio, 0.0001))
    
    if target_legits_count > max_rows:
        target_legits_count = max_rows
        
    legits = legits.sample(n=min(len(legits), target_legits_count), random_state=42)
    
    df_mini = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Keep only numeric columns to speed up river and avoid crashes
    numeric_cols = df_mini.select_dtypes(include=['number']).columns
    df_mini = df_mini[numeric_cols]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_mini.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with {len(df_mini)} rows (Frauds: {len(frauds)}). Original ratio: {ratio:.4f}, New ratio: {len(frauds)/len(df_mini):.4f}")

if __name__ == "__main__":
    create_mini_dataset(r"data\ULB Credit Card Fraud Detection\creditcard.csv", r"data\processed\ulb_mini.csv", "Class", max_rows=20000, max_frauds=100)
    create_mini_dataset(r"data\PaySim Synthetic Mobile Money Fraud\PS_20174392719_1491204439457_log.csv", r"data\processed\paysim_mini.csv", "isFraud", max_rows=20000, max_frauds=50)
    create_mini_dataset(r"data\IEEE-CIS\train_transaction.csv", r"data\processed\ieee_mini.csv", "isFraud", max_rows=20000, max_frauds=200)
    create_mini_dataset(r"data\banksim\bs140513_032310.csv", r"data\processed\banksim_mini.csv", "fraud", max_rows=20000, max_frauds=150)
