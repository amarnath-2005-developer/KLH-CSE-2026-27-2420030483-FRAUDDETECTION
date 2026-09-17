import pandas as pd
import numpy as np

def generate_behavioral_features(df: pd.DataFrame, entity_col: str = 'nameOrig') -> pd.DataFrame:
    """
    Generates cumulative transaction-level behavioral features for each unique entity.
    Calculated at the transaction level to prevent future leakage.
    
    Args:
        df (pd.DataFrame): The transaction dataset sorted by time.
        entity_col (str): The entity to group by (default 'nameOrig').
        
    Returns:
        pd.DataFrame: The dataframe with appended behavioral features.
    """
    print(f"Generating cumulative behavioral features for: '{entity_col}'...")
    
    # Ensure sorted by time
    df = df.sort_values(by=[entity_col, 'step']).copy()
    
    grouped = df.groupby(entity_col)['amount']
    
    # Cumulative Count (Transaction count up to this point)
    df['transaction_count'] = grouped.cumcount() + 1
    
    # Cumulative Sum
    df['cum_amount_sum'] = grouped.cumsum()
    
    # Cumulative Average
    df['average_amount'] = df['cum_amount_sum'] / df['transaction_count']
    
    # Cumulative Max and Min (using expanding)
    df['max_amount'] = grouped.expanding().max().reset_index(level=0, drop=True)
    df['min_amount'] = grouped.expanding().min().reset_index(level=0, drop=True)
    
    # Cumulative Variance (requires at least 2 observations)
    df['amount_variance'] = grouped.expanding().var().reset_index(level=0, drop=True)
    df['amount_variance'] = df['amount_variance'].fillna(0)
    
    # Transaction Velocity: cumulative amount / time active
    # time active = current step - very first step of this entity + 1
    first_step = df.groupby(entity_col)['step'].transform('min')
    df['time_active'] = (df['step'] - first_step) + 1
    df['transaction_velocity'] = df['cum_amount_sum'] / df['time_active']
    
    # Cleanup intermediate columns
    df = df.drop(columns=['cum_amount_sum', 'time_active'])
    
    for col in df.select_dtypes(include=['float64']).columns:
        if col not in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            df[col] = df[col].round(2)
            
    return df
