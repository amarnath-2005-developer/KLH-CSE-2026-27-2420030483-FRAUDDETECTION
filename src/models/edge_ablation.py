import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, confusion_matrix, f1_score
import numpy as np
import json
import time

root_dir = r"c:\Users\AMARNATH\OneDrive\Desktop\alt"
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Models & Utils
# ==========================================

class EdgeOnlyMLP(nn.Module):
    """ Ablation Model: Directly classifies using only a specific subset of edge attributes. """
    def __init__(self, edge_attr_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, edge_attr):
        return self.mlp(edge_attr).squeeze(-1)

def evaluate_metrics(y_true, y_pred_prob):
    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class present."}
        
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = f1_score(y_true, y_pred)
    
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(prec_val),
        "recall": float(rec_val),
        "f1": float(f1_val),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# ==========================================
# 2. Main Loop
# ==========================================

def run_ablation():
    print("--- Phase 6.3.1: GNN Signal Attribution Ablation ---")
    start_time = time.time()
    
    pt_path = os.path.join(root_dir, "data", "graph", "paysim_dev_graph.pt")
    dataset = torch.load(pt_path, weights_only=False)
    
    edge_attr = dataset['edge_attr']
    y = dataset['y'].float()
    train_mask = dataset['train_mask']
    test_mask = dataset['test_mask']
    feature_names = dataset['feature_names']
    
    print(f"Total features available: {len(feature_names)}")
    
    # Identify feature indices
    transaction_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest'
    ]
    # Add any 'type_' features dynamically
    transaction_features += [f for f in feature_names if f.startswith('type_')]
    
    behavior_features = [
        'origin_history_count', 'destination_history_count',
        'origin_history_avg_amount', 'destination_history_avg_amount',
        'origin_history_available', 'destination_history_available'
    ]
    
    mutation_features = [
        'origin_mutation', 'destination_mutation'
    ]
    
    # Filter indices exactly to what is present
    tx_idx = [feature_names.index(f) for f in transaction_features if f in feature_names]
    bh_idx = [feature_names.index(f) for f in behavior_features if f in feature_names]
    mu_idx = [feature_names.index(f) for f in mutation_features if f in feature_names]
    
    configs = {
        "A_Transaction_Only": tx_idx,
        "B_Transaction_Behavior": tx_idx + bh_idx,
        "C_Transaction_Mutation": tx_idx + mu_idx,
        "D_Transaction_Behavior_Mutation": tx_idx + bh_idx + mu_idx
    }
    
    results = {
        "metadata": {
            "test_transactions": int(test_mask.sum().item()),
            "test_fraud": int(y[test_mask].sum().item()),
            "epochs": 100,
            "seed": 42
        }
    }
    
    # Standardize ALL edge attributes based on train
    train_edge_attr = edge_attr[train_mask]
    attr_mean = train_edge_attr.mean(dim=0, keepdim=True)
    attr_std = train_edge_attr.std(dim=0, keepdim=True)
    attr_std[attr_std == 0] = 1.0 
    edge_attr_norm = (edge_attr - attr_mean) / attr_std
    
    # Class weights
    num_pos = y[train_mask].sum().item()
    num_neg = train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for config_name, indices in configs.items():
        print(f"\nEvaluating Configuration: {config_name} ({len(indices)} features)")
        
        # Subselect features
        train_features = edge_attr_norm[train_mask][:, indices]
        test_features = edge_attr_norm[test_mask][:, indices]
        
        # Reset seed before each model initialization for absolute consistency
        torch.manual_seed(42)
        model = EdgeOnlyMLP(edge_attr_dim=len(indices))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(100):
            model.train()
            opt.zero_grad()
            logits = model(train_features)
            loss = criterion(logits, y[train_mask])
            loss.backward()
            opt.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            logits = model(test_features)
            probs = torch.sigmoid(logits).numpy()
            
        y_test = y[test_mask].numpy()
        metrics = evaluate_metrics(y_test, probs)
        
        print(f"PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")
        
        results[config_name] = {
            "features_used": len(indices),
            "feature_names": [feature_names[i] for i in indices],
            "metrics": metrics
        }
        
    out_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "phase6_edge_ablation.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {res_path}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    run_ablation()
