import pandas as pd
import numpy as np
import argparse
import os

def generate_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates basic relationship (network) features from the transaction dataset.
    
    Supported Features based on PaySim Schema:
    - unique counterparties (Degree)
    - new counterparties (Temporal interaction discovery)
    - unique merchants (Based on 'M' prefix in nameDest)
    
    Unsupported Features (Not in Dataset):
    - unique devices: No IP, MAC, or Device ID column exists.
    - location/geospatial relationships: No location data exists.
    
    Args:
        df (pd.DataFrame): The raw transaction dataset.
        
    Returns:
        pd.DataFrame: A dataframe containing the network features appended.
    """
    print("Generating network relationship features...")
    
    # 1. Sort temporally to ensure logical "new counterparty" discovery
    df = df.sort_values(by=['step']).copy()
    
    # 2. Is Merchant Destination?
    # PaySim uses 'M' prefix for merchants in the nameDest column
    df['is_merchant_dest'] = df['nameDest'].str.startswith('M').astype(int)
    
    # 3. New Counterparty Discovery (Temporal Edge Creation)
    # We want to know if this is the first time 'nameOrig' is interacting with 'nameDest'
    print("Calculating temporal 'new counterparty' flags...")
    
    # We find the first occurrence of each (nameOrig, nameDest) pair
    # by dropping duplicates keeping the first occurrence (which is earliest due to sorting)
    first_interactions = df[['nameOrig', 'nameDest']].drop_duplicates(keep='first').copy()
    first_interactions['is_first_interaction'] = 1
    
    # Merge back to the main dataframe. 
    # To avoid marking every repeated interaction as "first", we need to align exactly by index.
    # A cleaner pandas way is to mark the duplicated rows. 
    # If a row is NOT a duplicate of the subset [nameOrig, nameDest], it's a new counterparty!
    df['is_new_counterparty'] = (~df.duplicated(subset=['nameOrig', 'nameDest'], keep='first')).astype(int)
    
    # 4. Unique Counterparties (Degree-like features per Originator)
    print("Calculating unique counterparty degree features...")
    
    # Cumulative unique counterparties interacted with up to the current transaction
    # We can do this by taking a cumulative sum of the 'is_new_counterparty' flag per 'nameOrig'
    df['cum_unique_destinations'] = df.groupby('nameOrig')['is_new_counterparty'].cumsum()
    
    # Cumulative unique merchants interacted with up to the current transaction
    # Only count if it's a new counterparty AND they are a merchant
    df['new_merchant_counterparty'] = df['is_new_counterparty'] & df['is_merchant_dest']
    df['cum_unique_merchants'] = df.groupby('nameOrig')['new_merchant_counterparty'].cumsum()
    
    # Drop intermediate columns
    df = df.drop(columns=['new_merchant_counterparty'])
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudDNA Network Feature Generator")
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
        
        # In PaySim, nameOrig rarely repeats in a small sample.
        # To prove the logic works, let's artificially duplicate a few rows 
        # so we can see 'is_new_counterparty' change from 1 to 0 for the same pair.
        print("\nInjecting a few artificial repeat transactions to demonstrate logic...")
        repeat_rows = df_sample.head(3).copy()
        repeat_rows['step'] += 1 # Make them happen later
        df_sample = pd.concat([df_sample, repeat_rows], ignore_index=True)
        
        # Run feature generation
        df_network = generate_network_features(df_sample)
        
        # Pick one of the duplicated entities to show the results
        test_entity = repeat_rows.iloc[0]['nameOrig']
        
        print("\n" + "="*80)
        print(f"DEMONSTRATION: Network Features for Entity '{test_entity}'")
        print("="*80)
        
        entity_data = df_network[df_network['nameOrig'] == test_entity]
        
        display_cols = [
            'step', 'nameOrig', 'nameDest', 'is_merchant_dest', 
            'is_new_counterparty', 'cum_unique_destinations'
        ]
        
        print(entity_data[display_cols].to_string(index=False))
        
        print("\nObservations:")
        print("1. 'is_merchant_dest' correctly identifies 'M' prefixed destinations.")
        print("2. 'is_new_counterparty' is 1 for the first interaction, and 0 for the repeated interaction.")
        print("3. 'cum_unique_destinations' (Degree) increments only on new counterparties.")

        print("\n--- Summary of Unavailable Features ---")
        print("- 'Unique Devices': Not generated. The dataset contains no device ID, IP, or MAC address.")
        print("- 'Geo-location': Not generated. The dataset contains no spatial data.")

    except Exception as e:
        print(f"Error during network feature generation: {e}")
