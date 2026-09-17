import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, confusion_matrix, f1_score
import numpy as np
import json
import time
from collections import defaultdict

root_dir = r"c:\Users\AMARNATH\OneDrive\Desktop\alt"
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ==========================================
# 1. Models & Utils
# ==========================================

class EdgeOnlyMLP(nn.Module):
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
    if len(y_true) == 0:
        return {"error": "Empty set"}
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

def run_verification():
    print("--- Phase 6.3.2: Neural Baseline Verification ---")
    start_time = time.time()
    
    pt_path = os.path.join(root_dir, "data", "graph", "paysim_dev_graph.pt")
    dataset = torch.load(pt_path, weights_only=False)
    
    edge_attr = dataset['edge_attr']
    y = dataset['y'].float()
    train_mask = dataset['train_mask']
    test_mask = dataset['test_mask']
    feature_names = dataset['feature_names']
    
    # 5. Feature Distribution Audit (Leakage-free, using entire dataset or train? Prompt asks for raw distributions of transaction features. Let's compute it over the whole dataset to show class separation, or just train set to be perfectly safe. We'll use train+test to show global properties of PaySim).
    print("\n[5] Feature Distribution Audit")
    raw_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    dist_stats = {}
    fraud_mask = (y == 1)
    nonfraud_mask = (y == 0)
    
    for f_name in raw_features:
        if f_name in feature_names:
            idx = feature_names.index(f_name)
            f_data = edge_attr[:, idx]
            dist_stats[f_name] = {
                "fraud_mean": float(f_data[fraud_mask].mean()),
                "fraud_std": float(f_data[fraud_mask].std()),
                "fraud_max": float(f_data[fraud_mask].max()),
                "nonfraud_mean": float(f_data[nonfraud_mask].mean()),
                "nonfraud_std": float(f_data[nonfraud_mask].std()),
                "nonfraud_max": float(f_data[nonfraud_mask].max())
            }
    
    # Identify feature indices
    transaction_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
    ]
    transaction_features += [f for f in feature_names if f.startswith('type_')]
    
    mutation_features = ['origin_mutation', 'destination_mutation']
    
    dest_context_features = [
        'destination_history_count', 'destination_history_avg_amount', 'destination_history_available'
    ]
    
    tx_idx = [feature_names.index(f) for f in transaction_features if f in feature_names]
    mu_idx = [feature_names.index(f) for f in mutation_features if f in feature_names]
    dc_idx = [feature_names.index(f) for f in dest_context_features if f in feature_names]
    all_idx = list(range(len(feature_names)))
    
    configs = {
        "Transaction_Only": tx_idx,
        "Transaction_Mutation": tx_idx + mu_idx,
        "Transaction_DestContext": tx_idx + dc_idx,
        "All_Features": all_idx
    }
    
    seeds = [42, 123, 2026]
    
    # Standardization (strictly on train)
    train_edge_attr = edge_attr[train_mask]
    attr_mean = train_edge_attr.mean(dim=0, keepdim=True)
    attr_std = train_edge_attr.std(dim=0, keepdim=True)
    attr_std[attr_std == 0] = 1.0 
    edge_attr_norm = (edge_attr - attr_mean) / attr_std
    
    # Class weights (strictly on train)
    num_pos = y[train_mask].sum().item()
    num_neg = train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    results = {
        "metadata": {
            "test_transactions": int(test_mask.sum().item()),
            "test_fraud": int(y[test_mask].sum().item()),
            "epochs": 100,
            "seeds": seeds
        },
        "feature_distribution_audit": dist_stats,
        "configurations": {},
        "conditional_analysis": defaultdict(lambda: defaultdict(dict)),
        "transaction_type_analysis": defaultdict(lambda: defaultdict(dict))
    }
    
    y_test = y[test_mask].numpy()
    
    # Subgroups for Conditional Analysis
    idx_orig_hist = feature_names.index('origin_history_count')
    idx_dest_avail = feature_names.index('destination_history_available')
    
    test_edge_attr = edge_attr[test_mask]
    mask_orig_avail = (test_edge_attr[:, idx_orig_hist] > 0).numpy()
    mask_orig_cold = (test_edge_attr[:, idx_orig_hist] == 0).numpy()
    mask_dest_avail = (test_edge_attr[:, idx_dest_avail] == 1).numpy()
    mask_dest_cold = (test_edge_attr[:, idx_dest_avail] == 0).numpy()
    
    # Subgroups for Transaction Type
    tx_masks = {}
    for tx_type in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
        f_name = f"type_{tx_type}"
        if f_name in feature_names:
            idx = feature_names.index(f_name)
            tx_masks[tx_type] = (test_edge_attr[:, idx] == 1).numpy()
            
    # Subgroup Metadata
    def get_subgroup_meta(mask):
        return {
            "sample_count": int(mask.sum()),
            "fraud_count": int(y_test[mask].sum()),
            "fraud_rate": float(y_test[mask].mean()) if mask.sum() > 0 else 0.0
        }
        
    results["subgroup_metadata"] = {
        "Group_A_Orig_Avail": get_subgroup_meta(mask_orig_avail),
        "Group_B_Orig_Cold": get_subgroup_meta(mask_orig_cold),
        "Group_C_Dest_Avail": get_subgroup_meta(mask_dest_avail),
        "Group_D_Dest_Cold": get_subgroup_meta(mask_dest_cold)
    }
    for k, v in tx_masks.items():
        results["subgroup_metadata"][f"Type_{k}"] = get_subgroup_meta(v)
    
    print("\n[2] Multi-Seed Verification")
    
    # Train & Evaluate
    for config_name, indices in configs.items():
        print(f"\nEvaluating: {config_name} ({len(indices)} features)")
        
        train_features = edge_attr_norm[train_mask][:, indices]
        test_features = edge_attr_norm[test_mask][:, indices]
        
        config_seed_metrics = []
        config_seed_probs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = EdgeOnlyMLP(edge_attr_dim=len(indices))
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(100):
                model.train()
                opt.zero_grad()
                logits = model(train_features)
                loss = criterion(logits, y[train_mask])
                loss.backward()
                opt.step()
                
            model.eval()
            with torch.no_grad():
                logits = model(test_features)
                probs = torch.sigmoid(logits).numpy()
                
            metrics = evaluate_metrics(y_test, probs)
            config_seed_metrics.append(metrics)
            config_seed_probs.append(probs)
            
        # 6. Stability Check
        pr_aucs = [m['pr_auc'] for m in config_seed_metrics]
        results["configurations"][config_name] = {
            "features": len(indices),
            "seed_results": config_seed_metrics,
            "stability": {
                "mean_pr_auc": float(np.mean(pr_aucs)),
                "std_pr_auc": float(np.std(pr_aucs)),
                "min_pr_auc": float(np.min(pr_aucs)),
                "max_pr_auc": float(np.max(pr_aucs))
            }
        }
        
        print(f"Mean PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
        
        # 3. Conditional Analysis (using the ensemble/mean probs across seeds, or just seed 42? Prompt implies evaluating models. We'll use the mean probability across seeds for conditional stability)
        mean_probs = np.mean(config_seed_probs, axis=0)
        
        results["conditional_analysis"]["Group_A_Orig_Avail"][config_name] = evaluate_metrics(y_test[mask_orig_avail], mean_probs[mask_orig_avail])
        results["conditional_analysis"]["Group_B_Orig_Cold"][config_name] = evaluate_metrics(y_test[mask_orig_cold], mean_probs[mask_orig_cold])
        results["conditional_analysis"]["Group_C_Dest_Avail"][config_name] = evaluate_metrics(y_test[mask_dest_avail], mean_probs[mask_dest_avail])
        results["conditional_analysis"]["Group_D_Dest_Cold"][config_name] = evaluate_metrics(y_test[mask_dest_cold], mean_probs[mask_dest_cold])
        
        # 4. Transaction-Type Analysis
        if config_name in ["Transaction_Only", "Transaction_Mutation", "All_Features"]:
            for tx_type, mask in tx_masks.items():
                res = evaluate_metrics(y_test[mask], mean_probs[mask])
                if "error" not in res:
                    results["transaction_type_analysis"][tx_type][config_name] = res["pr_auc"]
                else:
                    results["transaction_type_analysis"][tx_type][config_name] = None
        
    out_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "phase6_neural_verification.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {res_path}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    run_verification()
