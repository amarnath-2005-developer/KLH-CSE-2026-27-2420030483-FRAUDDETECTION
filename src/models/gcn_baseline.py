import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
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
# 1. Models
# ==========================================

class FraudGCN(nn.Module):
    """ Edge-Aware GCN model. Uses node structure + edge attributes. """
    def __init__(self, node_in_dim, edge_attr_dim, hidden_dim=16):
        super().__init__()
        # Message Passing (GCN)
        self.conv1 = geom_nn.GCNConv(node_in_dim, hidden_dim)
        self.conv2 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        
        # Edge Classification MLP
        # src_emb + dst_emb + edge_attr
        mlp_in = hidden_dim + hidden_dim + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, message_edge_index, eval_edge_index, edge_attr):
        """
        message_edge_index: strictly train edges for structural MP.
        eval_edge_index: the edges we want to predict on.
        """
        # Node embeddings
        h = self.conv1(x, message_edge_index)
        h = F.relu(h)
        h = self.conv2(h, message_edge_index)
        
        # Edge representations
        src, dst = eval_edge_index
        edge_rep = torch.cat([h[src], h[dst], edge_attr], dim=1)
        
        # Predict
        return self.mlp(edge_rep).squeeze(-1)

class EdgeOnlyMLP(nn.Module):
    """ Ablation Model: Directly classifies using only edge attributes. """
    def __init__(self, edge_attr_dim):
        super().__init__()
        # Same classifier structure but no node embeddings
        self.mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, edge_attr):
        return self.mlp(edge_attr).squeeze(-1)

# ==========================================
# 2. Evaluation Utilities
# ==========================================

def evaluate_metrics(y_true, y_pred_prob):
    # If all labels are the same (e.g. no fraud in a specific cut), metrics will fail
    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class present."}
        
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    # Binarize with 0.5 threshold
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
# 3. Main Loop
# ==========================================

def run_experiment():
    print("--- Phase 6.3: GNN Baseline & Ablation ---")
    start_time = time.time()
    
    # 1. Load Data
    pt_path = os.path.join(root_dir, "data", "graph", "paysim_dev_graph.pt")
    print(f"Loading {pt_path}...")
    dataset = torch.load(pt_path, weights_only=False)
    
    num_nodes = dataset['num_nodes']
    edge_index = dataset['edge_index']
    edge_attr = dataset['edge_attr']
    y = dataset['y'].float()
    train_mask = dataset['train_mask']
    test_mask = dataset['test_mask']
    feature_names = dataset['feature_names']
    
    orig_history_idx = feature_names.index('origin_history_count')
    is_cold_start = (edge_attr[:, orig_history_idx] == 0)
    
    dest_avail_idx = feature_names.index('destination_history_available')
    has_dest_hist = (edge_attr[:, dest_avail_idx] == 1)
    
    # Node features (constant 1.0)
    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    
    # Temporal Message Passing split
    train_edge_index = edge_index[:, train_mask]
    
    # Standardize edge attributes (mean=0, std=1) based ONLY on train
    train_edge_attr = edge_attr[train_mask]
    attr_mean = train_edge_attr.mean(dim=0, keepdim=True)
    attr_std = train_edge_attr.std(dim=0, keepdim=True)
    attr_std[attr_std == 0] = 1.0 # prevent div by zero
    
    edge_attr_norm = (edge_attr - attr_mean) / attr_std
    
    # Loss pos_weight from train
    num_pos = y[train_mask].sum().item()
    num_neg = train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    epochs = 100
    
    print("\n--- Training Model A: GCN + Edge Features ---")
    model_a = FraudGCN(node_in_dim=1, edge_attr_dim=17, hidden_dim=16)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    
    # Count params
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    print(f"Model A parameters: {params_a}")
    
    t_a = time.time()
    for epoch in range(epochs):
        model_a.train()
        opt_a.zero_grad()
        
        # Forward pass on train edges. 
        # message_edge_index = train_edge_index
        # eval_edge_index = train_edge_index (we only evaluate train edges for loss)
        logits = model_a(x, train_edge_index, train_edge_index, edge_attr_norm[train_mask])
        
        loss = criterion(logits, y[train_mask])
        loss.backward()
        opt_a.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    train_time_a = time.time() - t_a
    
    print("\n--- Training Model B: Edge-Only MLP (Ablation) ---")
    model_b = EdgeOnlyMLP(edge_attr_dim=17)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    
    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"Model B parameters: {params_b}")
    
    t_b = time.time()
    for epoch in range(epochs):
        model_b.train()
        opt_b.zero_grad()
        
        logits = model_b(edge_attr_norm[train_mask])
        loss = criterion(logits, y[train_mask])
        loss.backward()
        opt_b.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    train_time_b = time.time() - t_b
    
    print("\n--- Evaluation ---")
    
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        # Evaluate A (Test edges)
        # message_edge_index = train_edge_index (Strict temporal rule)
        # eval_edge_index = test_edge_index
        test_edge_index = edge_index[:, test_mask]
        
        logits_a = model_a(x, train_edge_index, test_edge_index, edge_attr_norm[test_mask])
        probs_a = torch.sigmoid(logits_a).numpy()
        
        # Evaluate B (Test edges)
        logits_b = model_b(edge_attr_norm[test_mask])
        probs_b = torch.sigmoid(logits_b).numpy()
        
    y_test = y[test_mask].numpy()
    
    overall_a = evaluate_metrics(y_test, probs_a)
    overall_b = evaluate_metrics(y_test, probs_b)
    
    # Cold Start Evaluation
    test_cold_start = is_cold_start[test_mask]
    cs_a = evaluate_metrics(y_test[test_cold_start], probs_a[test_cold_start])
    cs_b = evaluate_metrics(y_test[test_cold_start], probs_b[test_cold_start])
    
    # Destination Context Availability (among cold starts)
    test_cs_dest_avail = has_dest_hist[test_mask] & test_cold_start
    test_cs_dest_unavail = (~has_dest_hist)[test_mask] & test_cold_start
    
    da_a = evaluate_metrics(y_test[test_cs_dest_avail], probs_a[test_cs_dest_avail])
    du_a = evaluate_metrics(y_test[test_cs_dest_unavail], probs_a[test_cs_dest_unavail])
    
    # Build results
    results = {
        "metadata": {
            "model_a_params": params_a,
            "model_b_params": params_b,
            "train_time_a": train_time_a,
            "train_time_b": train_time_b,
            "test_transactions": int(test_mask.sum().item()),
            "test_fraud": int(y_test.sum())
        },
        "model_A_GCN": {
            "overall": overall_a,
            "cold_start_all": cs_a,
            "cold_start_dest_avail": da_a,
            "cold_start_dest_unavail": du_a
        },
        "model_B_MLP": {
            "overall": overall_b,
            "cold_start_all": cs_b
        }
    }
    
    out_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "gcn_baseline_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {res_path}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    run_experiment()
