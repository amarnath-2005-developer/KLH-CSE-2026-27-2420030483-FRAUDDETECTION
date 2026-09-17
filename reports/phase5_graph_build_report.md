# Phase 5.2: Temporal Graph Build Report

## 1. Overview
The Temporal Transaction Graph Builder successfully transformed the PaySim development dataset into a PyTorch Geometric compatible heterogeneous structure.

## 2. Graph Statistics
- **Nodes (Universal Entities)**: 9,073,900
- **Edges (Transactions)**: 6,362,620
- **Edge Features**: 17

### Temporal Split
- **Train Edges (`step <= 504`)**: 6,064,036 (Fraud: 5,621)
- **Test Edges (`step > 504`)**: 298,584 (Fraud: 2,592)

### Behavioral Demographics
- **Originator Cold-Start Edges**: 6,353,307 (99.85%)

## 3. Validation Results
- **Chronological Ordering Checked**: True (No future leakage possible in sequential traversal).
- **Train/Test Mutual Exclusivity**: True.
- **Cold-Start Definitions Preserved**: True (missing histories are correctly flagged by `origin_history_available`).

## 4. Feature Schema (`edge_attr`)
The following edge features are preserved and aligned per transaction:
```text
amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, origin_history_count, destination_history_count, origin_history_avg_amount, destination_history_avg_amount, origin_mutation, destination_mutation, origin_history_available, destination_history_available, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
```

## 5. System Metrics
- **Runtime**: 67.99 seconds
- **Output Size**: 770.91 MB
- **File**: `data/graph/paysim_dev_graph.npz`

*Note: The graph output format is a standard compressed numpy archive containing arrays for `edge_index`, `edge_attr`, `edge_time`, `y`, `train_mask`, and `test_mask`. This is universally compatible with DGL or PyG without forcing a dependency.*
