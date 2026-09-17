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
    
def compute_bootstrap_pr_auc(y_true, y_pred, n_iterations=1000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    aucs = []
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, n)
        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]
        
        if len(np.unique(y_true_b)) < 2:
            continue
            
        precision, recall, _ = precision_recall_curve(y_true_b, y_pred_b)
        aucs.append(auc(recall, precision))
        
    if len(aucs) == 0:
        return None, None
    
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return float(lower), float(upper)

# ==========================================
# 2. Main Loop
# ==========================================

def run_fullscale():
    print("--- Phase 7: Full-Scale FraudDNA Validation ---")
    start_time = time.time()
    
    npz_path = os.path.join(root_dir, "data", "graph", "paysim_full_graph.npz")
    print(f"Loading full graph dataset from {npz_path}...")
    dataset = np.load(npz_path, allow_pickle=True)
    
    edge_attr = dataset['edge_attr'].astype(np.float32)
    y = dataset['y'].astype(int)
    train_mask = dataset['train_mask']
    test_mask = dataset['test_mask']
    feature_names = dataset['feature_names'].tolist()
    edge_time = dataset['edge_time']
    
    # 9. Historical Context Coverage
    idx_orig_hist = feature_names.index('origin_history_available')
    idx_dest_hist = feature_names.index('destination_history_available')
    idx_orig_count = feature_names.index('origin_history_count')
    idx_dest_count = feature_names.index('destination_history_count')
    
    def get_coverage(mask):
        total = int(mask.sum())
        if total == 0: return {}
        orig_avail = int((edge_attr[mask, idx_orig_hist] == 1).sum())
        dest_avail = int((edge_attr[mask, idx_dest_hist] == 1).sum())
        return {
            "total": total,
            "origin_avail": orig_avail,
            "origin_avail_pct": (orig_avail / total) * 100,
            "dest_avail": dest_avail,
            "dest_avail_pct": (dest_avail / total) * 100
        }
        
    coverage = {
        "all": get_coverage(np.ones(len(y), dtype=bool)),
        "fraud": get_coverage(y == 1),
        "nonfraud": get_coverage(y == 0),
        "train": get_coverage(train_mask),
        "test": get_coverage(test_mask)
    }
    
    # 15. Feature Distribution Audit
    print("\n[15] Feature Distribution Audit")
    raw_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    dist_stats = {}
    fraud_mask = (y == 1)
    nonfraud_mask = (y == 0)
    
    for f_name in raw_features:
        if f_name in feature_names:
            idx = feature_names.index(f_name)
            f_data = edge_attr[:, idx]
            dist_stats[f_name] = {
                "fraud": {
                    "mean": float(f_data[fraud_mask].mean()),
                    "std": float(f_data[fraud_mask].std()),
                    "median": float(np.median(f_data[fraud_mask])),
                    "p90": float(np.percentile(f_data[fraud_mask], 90))
                },
                "nonfraud": {
                    "mean": float(f_data[nonfraud_mask].mean()),
                    "std": float(f_data[nonfraud_mask].std()),
                    "median": float(np.median(f_data[nonfraud_mask])),
                    "p90": float(np.percentile(f_data[nonfraud_mask], 90))
                }
            }
            
    # Configurations
    transaction_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    transaction_features += [f for f in feature_names if f.startswith('type_')]
    mutation_features = ['origin_mutation', 'destination_mutation']
    dest_context_features = ['destination_history_count', 'destination_history_avg_amount', 'destination_history_available']
    
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
    
    # 5. Leakage Controls (Strictly Train Mask)
    train_edge_attr = edge_attr[train_mask]
    attr_mean = train_edge_attr.mean(axis=0, keepdims=True)
    attr_std = train_edge_attr.std(axis=0, keepdims=True)
    attr_std[attr_std == 0] = 1.0 
    
    edge_attr_norm = (edge_attr - attr_mean) / attr_std
    edge_attr_norm = torch.tensor(edge_attr_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    num_pos = int(y[train_mask].sum())
    num_neg = int(train_mask.sum() - num_pos)
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    results = {
        "dataset_statistics": {
            "total_transactions": len(y),
            "train_transactions": int(train_mask.sum()),
            "test_transactions": int(test_mask.sum()),
            "total_fraud": int(y.sum()),
            "train_fraud": int(y[train_mask].sum()),
            "test_fraud": int(y[test_mask].sum()),
            "fraud_rate": float(y.mean()),
            "train_max_step": int(edge_time[train_mask].max()),
            "test_min_step": int(edge_time[test_mask].min())
        },
        "coverage": coverage,
        "feature_distribution_audit": dist_stats,
        "configurations": {},
        "conditional_analysis": defaultdict(lambda: defaultdict(dict)),
        "temporal_robustness": defaultdict(lambda: defaultdict(dict)),
        "transaction_type_analysis": defaultdict(lambda: defaultdict(dict)),
        "bootstrap": {}
    }
    
    # 10. Conditional Evaluation Subgroups (TEST SET ONLY)
    test_edge_attr = edge_attr[test_mask]
    y_test = y[test_mask]
    
    mask_orig_avail = (test_edge_attr[:, idx_orig_count] > 0)
    mask_orig_cold = (test_edge_attr[:, idx_orig_count] == 0)
    mask_dest_avail = (test_edge_attr[:, idx_dest_count] > 0)
    mask_dest_cold = (test_edge_attr[:, idx_dest_count] == 0)
    
    tx_masks = {}
    for tx_type in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
        f_name = f"type_{tx_type}"
        if f_name in feature_names:
            idx = feature_names.index(f_name)
            tx_masks[tx_type] = (test_edge_attr[:, idx] == 1)
            
    # Temporal Windows (TEST SET)
    test_time = edge_time[test_mask]
    window_masks = {
        "Window_1_505_584": (test_time >= 505) & (test_time <= 584),
        "Window_2_585_664": (test_time >= 585) & (test_time <= 664),
        "Window_3_665_744": (test_time >= 665) & (test_time <= 744)
    }
    
    seeds = [42, 123, 2026]
    mean_probs = {}
    
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
            
            # Since matrix is ~5M x 17 float32 (~340MB), we do full-batch training
            for epoch in range(100):
                model.train()
                opt.zero_grad()
                logits = model(train_features)
                loss = criterion(logits, y_tensor[train_mask])
                loss.backward()
                opt.step()
                
            model.eval()
            with torch.no_grad():
                logits = model(test_features)
                probs = torch.sigmoid(logits).numpy()
                
            metrics = evaluate_metrics(y_test, probs)
            config_seed_metrics.append(metrics)
            config_seed_probs.append(probs)
            
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
        mean_probs[config_name] = np.mean(config_seed_probs, axis=0)
        
        # Populate Conditional Analysis
        results["conditional_analysis"]["Group_A_Orig_Avail"][config_name] = evaluate_metrics(y_test[mask_orig_avail], mean_probs[config_name][mask_orig_avail])
        results["conditional_analysis"]["Group_B_Orig_Cold"][config_name] = evaluate_metrics(y_test[mask_orig_cold], mean_probs[config_name][mask_orig_cold])
        results["conditional_analysis"]["Group_C_Dest_Avail"][config_name] = evaluate_metrics(y_test[mask_dest_avail], mean_probs[config_name][mask_dest_avail])
        results["conditional_analysis"]["Group_D_Dest_Cold"][config_name] = evaluate_metrics(y_test[mask_dest_cold], mean_probs[config_name][mask_dest_cold])
        
        # Populate Temporal Robustness
        for w_name, w_mask in window_masks.items():
            results["temporal_robustness"][w_name][config_name] = evaluate_metrics(y_test[w_mask], mean_probs[config_name][w_mask])
            
        # Populate Transaction Types
        for t_name, t_mask in tx_masks.items():
            res = evaluate_metrics(y_test[t_mask], mean_probs[config_name][t_mask])
            results["transaction_type_analysis"][t_name][config_name] = res.get("pr_auc")
            
    # Subgroup Metadata injection
    def get_meta(mask):
        return {"sample_count": int(mask.sum()), "fraud_count": int(y_test[mask].sum())}
        
    for k in results["conditional_analysis"].keys():
        mask = eval(f"mask_{k.split('_')[2].lower()}_{k.split('_')[3].lower()}") if 'Orig' in k or 'Dest' in k else None # Hacky but safe here since we know the keys
        if 'Orig_Avail' in k: mask = mask_orig_avail
        if 'Orig_Cold' in k: mask = mask_orig_cold
        if 'Dest_Avail' in k: mask = mask_dest_avail
        if 'Dest_Cold' in k: mask = mask_dest_cold
        results["conditional_analysis"][k]["metadata"] = get_meta(mask)
        
    for k in window_masks:
        results["temporal_robustness"][k]["metadata"] = get_meta(window_masks[k])
        
    for k in tx_masks:
        results["transaction_type_analysis"][k]["metadata"] = get_meta(tx_masks[k])
        
    # 11. Incremental Signal (Delta)
    def calc_deltas(group_dict):
        try:
            base = group_dict["Transaction_Only"].get("pr_auc", 0)
            if base is None or "error" in group_dict["Transaction_Only"]: return {}
            return {
                "Delta_Mutation": group_dict["Transaction_Mutation"]["pr_auc"] - base,
                "Delta_DestContext": group_dict["Transaction_DestContext"]["pr_auc"] - base,
                "Delta_All": group_dict["All_Features"]["pr_auc"] - base
            }
        except:
            return {}
            
    results["incremental_signal"] = {
        "Overall": calc_deltas({"Transaction_Only": {"pr_auc": results["configurations"]["Transaction_Only"]["stability"]["mean_pr_auc"]},
                                "Transaction_Mutation": {"pr_auc": results["configurations"]["Transaction_Mutation"]["stability"]["mean_pr_auc"]},
                                "Transaction_DestContext": {"pr_auc": results["configurations"]["Transaction_DestContext"]["stability"]["mean_pr_auc"]},
                                "All_Features": {"pr_auc": results["configurations"]["All_Features"]["stability"]["mean_pr_auc"]}}),
        "Group_C_Dest_Avail": calc_deltas(results["conditional_analysis"]["Group_C_Dest_Avail"]),
        "Group_D_Dest_Cold": calc_deltas(results["conditional_analysis"]["Group_D_Dest_Cold"])
    }
    
    # 14. Statistical Confidence (Bootstrap Group C)
    print("\n[14] Bootstrapping Group C...")
    y_test_c = y_test[mask_dest_avail]
    for config in ["Transaction_Only", "Transaction_Mutation", "Transaction_DestContext"]:
        pred_c = mean_probs[config][mask_dest_avail]
        lower, upper = compute_bootstrap_pr_auc(y_test_c, pred_c)
        results["bootstrap"][config] = {
            "lower_95": lower,
            "upper_95": upper,
            "point_estimate": results["conditional_analysis"]["Group_C_Dest_Avail"][config].get("pr_auc")
        }
    
    out_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "phase7_fullscale_validation.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {res_path}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    run_fullscale()
