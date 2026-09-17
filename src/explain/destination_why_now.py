import pandas as pd
import numpy as np
import sys
import os

# Ensure src is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

def explain_destination_mutation(row: pd.Series) -> str:
    """
    Generates a human-readable 'WHY-NOW' explanation for a destination's behavioral mutation.
    Strictly uses actual measured feature differences without assuming causality or claiming fraud.
    """
    dest = row.get('nameDest', 'Unknown')
    score = row.get('mutation_score', 0.0)
    
    # If no history, it's a cold start.
    if row.get('has_sufficient_history', 0) == 0:
        return f"Destination: {dest}\nBehavioral changes: No historical baseline available. Cold-start entity.\n"
        
    # Extract historical metrics
    h_count = int(row.get('hist_incoming_txn_count', 0))
    h_avg = row.get('hist_incoming_avg_amount', 0.0)
    h_new_orig = int(row.get('hist_new_originators_count', 0))
    h_vel = row.get('hist_incoming_velocity', 0.0)
    h_burst = int(row.get('hist_burst_count', 0))
    
    # Extract current metrics
    c_count = int(row.get('curr_incoming_txn_count', 0))
    c_avg = row.get('curr_incoming_avg_amount', 0.0)
    c_new_orig = int(row.get('curr_new_originators_count', 0))
    c_vel = row.get('curr_incoming_velocity', 0.0)
    c_burst = int(row.get('curr_burst_count', 0))
    
    # Determine the significant behavioral changes driving the score
    changes = []
    
    if c_vel > (h_vel * 2) and c_vel > 100:
        changes.append("incoming velocity increased significantly")
    
    if c_avg > (h_avg * 2) and c_avg > 1000:
        changes.append("amount scale shifted higher")
    elif c_avg < (h_avg * 0.5):
        changes.append("amount scale shifted lower")
        
    if c_new_orig > h_new_orig:
        changes.append("new originators appeared")
        
    if c_burst > h_burst:
        changes.append("transaction burst detected")
        
    if c_count > (h_count * 2):
        changes.append("incoming transaction frequency increased")
        
    if not changes:
        if score > 0.3:
            changes.append("moderate variances across multiple dimensions contributed to the score")
        else:
            changes.append("minor fluctuations within baseline variance")
            
    # Format the explanation precisely as requested
    explanation = f"Destination: {dest}\n"
    explanation += f"Mutation Score: {score:.4f}\n\n"
    
    explanation += "Historical:\n"
    explanation += f"Incoming transactions: {h_count}\n"
    explanation += f"Average amount: {h_avg:,.2f}\n"
    explanation += f"Unique/New originators: {h_new_orig}\n"
    explanation += f"Velocity: {h_vel:,.2f}\n\n"
    
    explanation += "Current:\n"
    explanation += f"Incoming transactions: {c_count}\n"
    explanation += f"Average amount: {c_avg:,.2f}\n"
    explanation += f"Unique/New originators: {c_new_orig}\n"
    explanation += f"Velocity: {c_vel:,.2f}\n\n"
    
    explanation += "Behavioral changes:\n"
    for change in changes:
        explanation += f"- {change}\n"
        
    return explanation

if __name__ == "__main__":
    from src.fraud_dna.destination_fingerprint import generate_destination_fingerprints
    from src.fraud_dna.mutation import calculate_mutation_score

    print("Loading dataset for WHY-NOW explanation test...")
    data_path = os.path.join(root_dir, "data", "PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(data_path)
    
    print("Sampling entities...")
    # Get a mix of fraud and normal entities to ensure we have valid history to explain
    fraud_dest = df[df['isFraud'] == 1]['nameDest'].unique()
    np.random.seed(42)
    sample_normal = np.random.choice(df[df['isFraud'] == 0]['nameDest'].unique(), size=5000, replace=False)
    
    df_sample = df[df['nameDest'].isin(set(fraud_dest).union(set(sample_normal)))].copy()
    
    print("Running Pipeline...")
    df_fp = generate_destination_fingerprints(df_sample, hist_window=168, curr_window=24)
    df_scored = calculate_mutation_score(
        df_fp, 
        features=['incoming_txn_count', 'incoming_avg_amount', 'incoming_amount_std', 'incoming_velocity', 'new_originators_count', 'burst_count'], 
        method='symmetric_relative'
    )
    
    valid_df = df_scored[df_scored['has_sufficient_history'] == 1]
    
    print("\n" + "="*80)
    print("EXAMPLE: HIGH MUTATION (SUSPICIOUS/FRAUD)")
    print("="*80)
    high_mut = valid_df[(valid_df['mutation_score'] > 0.7) & (valid_df['isFraud'] == 1)]
    if not high_mut.empty:
        example = high_mut.iloc[0]
        print(explain_destination_mutation(example))
    
    print("\n" + "="*80)
    print("EXAMPLE: LOW/MODERATE MUTATION (NORMAL)")
    print("="*80)
    low_mut = valid_df[(valid_df['mutation_score'] > 0.1) & (valid_df['mutation_score'] < 0.4) & (valid_df['isFraud'] == 0)]
    if not low_mut.empty:
        example2 = low_mut.iloc[0]
        print(explain_destination_mutation(example2))
