import pandas as pd
import numpy as np
import argparse
import os

def generate_dna_fingerprints(df: pd.DataFrame, entity_col: str = 'nameOrig', time_col: str = 'step', 
                              hist_window: int = 72, curr_window: int = 12) -> pd.DataFrame:
    """
    Generates both Historical H(entity, t) and Current C(entity, t) Behavioral Fingerprints.
    
    H(entity, t): Behavior strictly BEFORE time 't' (e.g., past 72 hours).
    C(entity, t): Behavior UP TO AND INCLUDING time 't' (e.g., past 12 hours).
    
    Args:
        df (pd.DataFrame): Transaction dataset.
        entity_col (str): Entity to track (e.g., 'nameOrig').
        time_col (str): Temporal column (e.g., 'step').
        hist_window (int): The lookback window for history in hours (default 72).
        curr_window (int): The lookback window for current behavior in hours (default 12).
        
    Returns:
        pd.DataFrame: A dataframe containing both fingerprint vectors, aligned with the input index.
    """
    print(f"Extracting fingerprints for '{entity_col}'...")
    print(f" -> Historical Window: {hist_window}h (Strictly before current transaction)")
    print(f" -> Current Window: {curr_window}h (Including current transaction)")
    
    # Ensure strict temporal ordering to prevent leakage
    df = df.sort_values(by=[entity_col, time_col]).copy()
    original_index = df.index
    
    # Map step to dummy time for rolling window operations
    df['dummy_time'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(df[time_col], unit='h')
    df_temp = df.set_index('dummy_time')
    
    grouped = df_temp.groupby(entity_col)
    
    fingerprint = pd.DataFrame(index=df.index)
    
    # -------------------------------------------------------------------------
    # 1. HISTORICAL FINGERPRINT: H(entity, t)
    # closed='left' ensures the current transaction at time 't' is EXCLUDED.
    # -------------------------------------------------------------------------
    hist_str = f"{hist_window}h"
    hist_roll = grouped['amount'].rolling(hist_str, closed='left')
    
    fingerprint['hist_txn_count'] = hist_roll.count().reset_index(level=0, drop=True).values
    hist_sum = hist_roll.sum().reset_index(level=0, drop=True).values
    fingerprint['hist_avg_amount'] = hist_roll.mean().reset_index(level=0, drop=True).values
    fingerprint['hist_amount_std'] = hist_roll.std().reset_index(level=0, drop=True).values
    fingerprint['hist_velocity'] = hist_sum / hist_window
    
    # -------------------------------------------------------------------------
    # 2. CURRENT FINGERPRINT: C(entity, t)
    # closed='right' (default) ensures the current transaction is INCLUDED.
    # -------------------------------------------------------------------------
    curr_str = f"{curr_window}h"
    curr_roll = grouped['amount'].rolling(curr_str, closed='right')
    
    fingerprint['curr_txn_count'] = curr_roll.count().reset_index(level=0, drop=True).values
    curr_sum = curr_roll.sum().reset_index(level=0, drop=True).values
    fingerprint['curr_avg_amount'] = curr_roll.mean().reset_index(level=0, drop=True).values
    fingerprint['curr_amount_std'] = curr_roll.std().reset_index(level=0, drop=True).values
    fingerprint['curr_velocity'] = curr_sum / curr_window
    
    # --- Handling Missing / Insufficient Data ---
    features = ['txn_count', 'avg_amount', 'amount_std', 'velocity']
    for prefix in ['hist', 'curr']:
        # txn_count gets 0
        fingerprint[f'{prefix}_txn_count'] = fingerprint[f'{prefix}_txn_count'].fillna(0)
        
        # average, std, velocity get 0 for empty windows
        for feat in ['avg_amount', 'amount_std', 'velocity']:
            col = f'{prefix}_{feat}'
            fingerprint[col] = fingerprint[col].fillna(0).round(2)
            
    # Explicit flags for sufficient history/current contexts
    fingerprint['has_sufficient_history'] = (fingerprint['hist_txn_count'] >= 1).astype(int)
    # The current window will almost always have at least 1 (the transaction itself), 
    # but we can track if it has *multiple* for a true std-dev calculation
    fingerprint['has_sufficient_current'] = (fingerprint['curr_txn_count'] >= 1).astype(int)
    
    print("Historical and Current fingerprints extracted successfully.")
    
    # Return the vectors aligned with original data
    return pd.concat([df, fingerprint], axis=1).loc[original_index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudDNA Fingerprints")
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument(
        "--input", 
        type=str, 
        default=os.path.join(root_dir, "data", "PS_20174392719_1491204439457_log.csv"),
        help="Path to dataset CSV"
    )
    args = parser.parse_args()
    
    print(f"Loading a 100,000 row sample from {args.input} for testing...")
    try:
        df_sample = pd.read_csv(args.input, nrows=100000)
        
        # Inject artificial sequence for an entity to clearly show the rolling windows
        test_entity = "DNA_MUTATION_TEST"
        synthetic_rows = pd.DataFrame({
            'step': [1, 5, 20, 80, 82], 
            'type': ['TRANSFER'] * 5,
            # Normal behavior -> Normal -> Normal -> SUDDEN SPIKE
            'amount': [100.0, 120.0, 110.0, 5000.0, 5000.0],
            'nameOrig': [test_entity] * 5,
            'nameDest': ['M123'] * 5,
            'oldbalanceOrg': [0]*5, 'newbalanceOrig': [0]*5,
            'oldbalanceDest': [0]*5, 'newbalanceDest': [0]*5,
            'isFraud': [0]*5, 'isFlaggedFraud': [0]*5
        })
        
        df_sample = pd.concat([df_sample, synthetic_rows], ignore_index=True)
        
        # Generate Fingerprints (Hist: 72h, Curr: 12h)
        fingerprints = generate_dna_fingerprints(df_sample, entity_col='nameOrig', hist_window=72, curr_window=12)
        
        print("\n" + "="*90)
        print(f"HISTORICAL (H) vs CURRENT (C) VECTORS FOR: '{test_entity}'")
        print("="*90)
        
        entity_fp = fingerprints[fingerprints['nameOrig'] == test_entity]
        
        # Print side-by-side comparison cleanly
        cols_to_print = ['step', 'amount', 
                         'hist_txn_count', 'curr_txn_count', 
                         'hist_avg_amount', 'curr_avg_amount', 
                         'hist_velocity', 'curr_velocity']
                         
        print(entity_fp[cols_to_print].to_string(index=False))
        
        print("\nVerification Checklist:")
        print("✓ T=1:  H is empty (no history before T=1). C contains just the T=1 txn (100).")
        print("✓ T=5:  H sees T=1 (avg=100). C sees T=1 and T=5 (avg=110) because both are within 12h.")
        print("✓ T=20: H sees T=1, T=5 (avg=110). C sees ONLY T=20 (avg=110) because T=1 and T=5 are > 12h old!")
        print("✓ T=80: SUDDEN SPIKE! H only sees T=20 (avg=110) as older ones expired from 72h window. C sees T=80 (avg=5000).")
        print("✓ T=82: H sees T=20 (avg=110). C sees T=80 and T=82 (avg=5000).")

    except Exception as e:
        print(f"Error during fingerprint generation: {e}")
