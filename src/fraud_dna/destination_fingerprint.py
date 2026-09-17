import pandas as pd
import numpy as np
import os
import sys

# Ensure src is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.fraud_dna.mutation import calculate_mutation_score

def generate_destination_fingerprints(df: pd.DataFrame, hist_window: int = 168, curr_window: int = 24) -> pd.DataFrame:
    """
    Generates Historical and Current Behavioral Fingerprints focused entirely on INCOMING
    traffic to the Destination Entity (nameDest).
    """
    print(f"Extracting DESTINATION fingerprints over Hist:{hist_window}h / Curr:{curr_window}h...")
    
    # 1. Sort strictly chronologically by destination
    df = df.sort_values(by=['nameDest', 'step']).copy()
    original_index = df.index
    
    # 2. Transaction-level pre-calculations
    # New Originator Flag
    df['is_new_originator'] = (~df.duplicated(subset=['nameDest', 'nameOrig'], keep='first')).astype(float)
    
    # Time Gap (Time since this destination last received a transaction)
    df['prev_step'] = df.groupby('nameDest')['step'].shift(1)
    df['time_gap'] = df['step'] - df['prev_step']
    df['time_gap'] = df['time_gap'].fillna(-1.0)
    
    # Burst indicator (Rapid successive incoming transactions in the exact same hour)
    df['is_burst'] = (df['time_gap'] == 0).astype(float)
    
    # 3. Dummy Time for rolling windows
    df['dummy_time'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(df['step'], unit='h')
    df_temp = df.set_index('dummy_time')
    grouped = df_temp.groupby('nameDest')
    
    fingerprint = pd.DataFrame(index=df.index)
    
    hist_str = f"{hist_window}h"
    curr_str = f"{curr_window}h"
    
    # Historical (closed='left')
    print("Computing Historical Aggregations...")
    hist_roll_amt = grouped['amount'].rolling(hist_str, closed='left')
    hist_roll_new_orig = grouped['is_new_originator'].rolling(hist_str, closed='left')
    hist_roll_gap = grouped['time_gap'].rolling(hist_str, closed='left')
    hist_roll_burst = grouped['is_burst'].rolling(hist_str, closed='left')
    
    fingerprint['hist_incoming_txn_count'] = hist_roll_amt.count().reset_index(level=0, drop=True).values
    hist_sum = hist_roll_amt.sum().reset_index(level=0, drop=True).values
    fingerprint['hist_incoming_avg_amount'] = hist_roll_amt.mean().reset_index(level=0, drop=True).values
    fingerprint['hist_incoming_amount_std'] = hist_roll_amt.std().reset_index(level=0, drop=True).values
    fingerprint['hist_incoming_velocity'] = hist_sum / hist_window
    fingerprint['hist_new_originators_count'] = hist_roll_new_orig.sum().reset_index(level=0, drop=True).values
    fingerprint['hist_avg_time_gap'] = hist_roll_gap.mean().reset_index(level=0, drop=True).values
    fingerprint['hist_burst_count'] = hist_roll_burst.sum().reset_index(level=0, drop=True).values
    
    # Current (closed='right')
    print("Computing Current Aggregations...")
    curr_roll_amt = grouped['amount'].rolling(curr_str, closed='right')
    curr_roll_new_orig = grouped['is_new_originator'].rolling(curr_str, closed='right')
    curr_roll_gap = grouped['time_gap'].rolling(curr_str, closed='right')
    curr_roll_burst = grouped['is_burst'].rolling(curr_str, closed='right')
    
    fingerprint['curr_incoming_txn_count'] = curr_roll_amt.count().reset_index(level=0, drop=True).values
    curr_sum = curr_roll_amt.sum().reset_index(level=0, drop=True).values
    fingerprint['curr_incoming_avg_amount'] = curr_roll_amt.mean().reset_index(level=0, drop=True).values
    fingerprint['curr_incoming_amount_std'] = curr_roll_amt.std().reset_index(level=0, drop=True).values
    fingerprint['curr_incoming_velocity'] = curr_sum / curr_window
    fingerprint['curr_new_originators_count'] = curr_roll_new_orig.sum().reset_index(level=0, drop=True).values
    fingerprint['curr_avg_time_gap'] = curr_roll_gap.mean().reset_index(level=0, drop=True).values
    fingerprint['curr_burst_count'] = curr_roll_burst.sum().reset_index(level=0, drop=True).values
    
    # Clean up NaNs
    features = [
        'incoming_txn_count', 'incoming_avg_amount', 'incoming_amount_std', 
        'incoming_velocity', 'new_originators_count', 'avg_time_gap', 'burst_count'
    ]
    
    for prefix in ['hist', 'curr']:
        for feat in features:
            col = f'{prefix}_{feat}'
            fingerprint[col] = fingerprint[col].fillna(0).round(2)
            
    fingerprint['has_sufficient_history'] = (fingerprint['hist_incoming_txn_count'] >= 1).astype(int)
    
    return pd.concat([df, fingerprint], axis=1).loc[original_index]

if __name__ == "__main__":
    print("Running destination fingerprint test...")
    # Will run via analysis script.
