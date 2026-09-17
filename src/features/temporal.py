import pandas as pd
import numpy as np
import argparse
import os

def generate_temporal_features(df: pd.DataFrame, entity_col: str = 'nameDest') -> pd.DataFrame:
    """
    Generates temporal behavioral features for transactions.
    Calculations strictly respect time ordering (step) and never peek into the future.
    
    Args:
        df (pd.DataFrame): The raw transaction dataset containing 'step'.
        entity_col (str): The column representing the entity to group by (default 'nameDest').
        
    Returns:
        pd.DataFrame: A dataframe containing the new temporal features aligned with the original index.
    """
    print(f"Generating temporal features grouped by entity: '{entity_col}'...")
    
    # 1. Sort strictly by time to ensure no future leakage
    df = df.sort_values(by=[entity_col, 'step']).copy()
    
    # 2. Extract cyclical time features (PaySim 'step' is 1 hour of time)
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7
    
    # 3. Calculate Recency / Time Since Previous Transaction
    # Using shift(1) per entity strictly looks at the *past* row only.
    df['prev_step'] = df.groupby(entity_col)['step'].shift(1)
    df['time_since_last_txn'] = df['step'] - df['prev_step']
    
    # Fill NaNs for the very first transaction of an entity with -1 (or a large number)
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(-1)
    
    # 4. Recent Transaction Count (Rolling Window)
    # To do time-based rolling in pandas, we need a datetime-like column.
    # We will map 'step' (hours) to a dummy datetime.
    df['dummy_time'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(df['step'], unit='h')
    
    # We set the index to the dummy_time for rolling operations
    df_temp = df.set_index('dummy_time')
    
    # Rolling 24-hour count. 
    # closed='left' ensures the current transaction is NOT counted (strict historical lookback).
    print("Calculating rolling 24-hour historical transaction count...")
    rolling_counts = (
        df_temp.groupby(entity_col)['step']
        .rolling('24h', closed='left')
        .count()
    )
    
    # Realign the rolling counts back to the main dataframe
    df['recent_txn_count_24h'] = rolling_counts.values
    df['recent_txn_count_24h'] = df['recent_txn_count_24h'].fillna(0)
    
    # 5. Transaction Burst Indicator
    # A burst is flagged if multiple transactions happen in the same step (0 hours since last) 
    # OR if they have more than 5 transactions in the last 24 historical hours.
    df['is_burst_activity'] = ((df['time_since_last_txn'] == 0) | (df['recent_txn_count_24h'] > 5)).astype(int)
    
    # Clean up intermediate columns
    df = df.drop(columns=['prev_step', 'dummy_time'])
    
    # Return to original sort order if needed, but keeping time-sorted is best practice.
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudDNA Temporal Feature Generator")
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument(
        "--input", 
        type=str, 
        default=os.path.join(root_dir, "data", "PS_20174392719_1491204439457_log.csv"),
        help="Path to dataset CSV"
    )
    args = parser.parse_args()
    
    print(f"Loading a 100,000 row sample from: {args.input} for testing...")
    try:
        # Load sample
        df_sample = pd.read_csv(args.input, nrows=100000)
        
        # We will use 'nameDest' as merchants/hubs have repeated temporal activity
        entity_col = 'nameDest'
        df_temporal = generate_temporal_features(df_sample, entity_col=entity_col)
        
        # Find an entity that has multiple transactions over time to demonstrate strict historical rolling
        counts = df_temporal[entity_col].value_counts()
        test_entity = counts.index[0] # The entity with the most transactions in the sample
        
        print("\n" + "="*80)
        print(f"DEMONSTRATION: Temporal Ordering and No Future Leakage for Entity '{test_entity}'")
        print("="*80)
        
        entity_data = df_temporal[df_temporal[entity_col] == test_entity]
        
        # Select columns to display the progression over time clearly
        display_cols = [
            'step', 'hour_of_day', 'day_of_week', 
            'time_since_last_txn', 'recent_txn_count_24h', 'is_burst_activity'
        ]
        
        print(entity_data[display_cols].head(10).to_string(index=False))
        
        print("\nObservations ensuring NO FUTURE LEAKAGE:")
        print("1. 'time_since_last_txn' for the very first step is -1 (no past exists).")
        print("2. 'recent_txn_count_24h' only counts transactions that occurred STRICTLY BEFORE the current step (using closed='left').")
        print("3. When a transaction happens at the exact same 'step', 'time_since_last_txn' is 0, triggering the 'is_burst_activity' flag.")

    except Exception as e:
        print(f"Error during temporal feature generation: {e}")
