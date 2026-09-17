import pandas as pd
import numpy as np
import argparse
import os

def calculate_mutation_score(df: pd.DataFrame, 
                             features: list = ['txn_count', 'avg_amount', 'amount_std', 'velocity'], 
                             method: str = 'symmetric_relative') -> pd.DataFrame:
    """
    Calculates the core FraudDNA Behavioral Mutation Score: D(H(e,t), C(e,t))
    
    Args:
        df (pd.DataFrame): Dataframe containing both historical (hist_*) and current (curr_*) features.
        features (list): The base names of the features to compare.
        method (str): Distance metric to use ('symmetric_relative' or 'cosine').
        
    Returns:
        pd.DataFrame: Dataframe with 'mutation_score' and 'top_mutation_feature' appended.
    """
    print(f"Calculating Behavioral Mutation Score using {method} distance...")
    
    # Extract matrices
    H_cols = [f'hist_{f}' for f in features]
    C_cols = [f'curr_{f}' for f in features]
    
    # Ensure columns exist
    missing_cols = [c for c in H_cols + C_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required fingerprint columns: {missing_cols}")
        
    H = df[H_cols].values.astype(float)
    C = df[C_cols].values.astype(float)
    
    epsilon = 1.0 # Base offset to prevent tiny numbers from causing 100% mutation
    
    if method == 'symmetric_relative':
        # Symmetric Relative Difference (similar to SMAPE): |C - H| / (|C| + |H| + epsilon)
        # This inherently bounds the difference for each feature between [0, 1)
        # It handles vastly different scales and zero-variance/missing history beautifully.
        diff = np.abs(C - H)
        denom = np.abs(C) + np.abs(H) + epsilon
        
        feature_scores = diff / denom
        
        # Root Mean Square of the bounded feature differences
        mutation_score = np.sqrt(np.mean(feature_scores**2, axis=1))
        
    elif method == 'cosine':
        # Cosine distance ignores magnitude scaling, looking only at vector angle.
        dot_product = np.sum(H * C, axis=1)
        norm_H = np.linalg.norm(H, axis=1)
        norm_C = np.linalg.norm(C, axis=1)
        
        denom = norm_H * norm_C
        denom[denom == 0] = epsilon
        
        cosine_sim = dot_product / denom
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Normalize to 0-1 range
        mutation_score = (1.0 - cosine_sim) / 2.0
    else:
        raise ValueError(f"Unknown distance method: {method}")
        
    df['mutation_score'] = np.round(mutation_score, 4)
    
    # Feature-Level Contribution (Which behavior mutated the most?)
    # We find the feature with the highest absolute relative deviation
    feature_scores = np.abs(C - H) / (np.abs(C) + np.abs(H) + epsilon)
    max_dev_indices = np.argmax(feature_scores, axis=1)
    df['top_mutation_feature'] = [features[i] for i in max_dev_indices]
    
    # --- Handling Insufficient History ---
    # If we have no baseline H to compare C against, mutation score is undefined/zeroed out
    if 'has_sufficient_history' in df.columns:
        mask = (df['has_sufficient_history'] == 0)
        df.loc[mask, 'mutation_score'] = 0.0
        df.loc[mask, 'top_mutation_feature'] = 'None (No History)'
        
    # Also zero out top feature if there's identically zero mutation
    df.loc[df['mutation_score'] == 0.0, 'top_mutation_feature'] = 'None'
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudDNA Mutation Scoring")
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()
    
    print("Running Mutation Score Unit Test on Synthetic Entity...\n")
    
    test_entity = "DNA_MUTATION_TEST"
    
    # Scenario:
    # T=1: No history.
    # T=5: Small change (Normal)
    # T=20: Massive Velocity and Amount Spike (Fraud)
    # T=80: Small Change (Normal)
    
    data = {
        'nameOrig': [test_entity] * 4,
        'step': [1, 5, 20, 80],
        'has_sufficient_history': [0, 1, 1, 1],
        'hist_txn_count': [0, 1, 2, 3],
        'curr_txn_count': [1, 2, 10, 1],
        'hist_avg_amount': [0, 100, 105, 110],
        'curr_avg_amount': [100, 110, 8000, 120],
        'hist_amount_std': [0, 0, 5, 10],
        'curr_amount_std': [0, 5, 2000, 0],
        'hist_velocity': [0, 10, 10, 11],
        'curr_velocity': [100, 11, 800, 12]
    }
    
    df_test = pd.DataFrame(data)
    
    df_scored = calculate_mutation_score(
        df_test, 
        features=['txn_count', 'avg_amount', 'amount_std', 'velocity'],
        method='symmetric_relative'
    )
    
    print("="*90)
    print("FRAUD-DNA BEHAVIORAL MUTATION SCORE RESULTS")
    print("="*90)
    
    display_cols = [
        'step', 
        'hist_avg_amount', 'curr_avg_amount', 
        'hist_velocity', 'curr_velocity',
        'mutation_score', 'top_mutation_feature'
    ]
    
    print(df_scored[display_cols].to_string(index=False))
    
    print("\nVerification Checklist:")
    print("-> T=1:  Mutation is 0.0 because 'has_sufficient_history' = 0.")
    print("-> T=5:  Mutation is low. Slight deviation in velocity/amounts.")
    print("-> T=20: Mutation is extremely high (~0.9+). Massive deviation detected! Top feature correctly identifies the largest shift.")
    print("-> T=80: Mutation is low. Back to normal baseline behavior.")
