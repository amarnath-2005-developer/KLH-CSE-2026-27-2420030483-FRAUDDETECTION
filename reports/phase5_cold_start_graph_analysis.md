# Phase 5.5: Cold-Start Graph Analysis

## 1. Overview
This report evaluates the availability of destination behavioral context for originator cold-start transactions within the PyTorch Geometric compatible graph. The analysis is purely descriptive and strictly enforces temporal rules.

- **Total Transactions**: 1,219,678
- **Total Fraud**: 8,213

## 2. Originator Cold-Start Demographics
An originator cold-start is strictly defined as `origin_history_count == 0` prior to the transaction.

- **Cold-Start Transactions**: 1,218,470 (99.90%)
- **Cold-Start Fraud**: 8,212 (99.99% of all fraud)

## 3. Destination Context on Cold-Start Edges
Among the 1,218,470 cold-start transactions:
- **Destination History Available**: 507,835 (41.68%)
- **Destination History Unavailable**: 710,635
- **Destination Mutation Available**: 507,835
- **Destination Mutation Unavailable**: 710,635

### History Depth Distribution (When Available)
For cold-start edges connecting to a destination with existing history:
- **Mean History Count**: 5.74
- **Median History Count**: 3.00
- **Minimum**: 1
- **Maximum**: 112
- **25th / 75th / 90th Percentiles**: 1 / 7 / 15

## 4. Comparison: With vs Without Destination Context
*Warning: Higher fraud rates in historically rich destinations do not imply destination context causes fraud; it merely highlights network clustering patterns.*

| Metric | With Destination Context | Without Destination Context |
|--------|--------------------------|-----------------------------|
| **Transaction Count** | 507,835 | 710,635 |
| **Fraud Count** | 1,625 | 6,587 |
| **Fraud Rate** | 0.3200% | 0.9269% |
| **Mean Destination Depth** | 5.74 | 0.00 |

## 5. Temporal Breakdown (Train vs Test)
Chronological split enforced exactly at `step 504`.

### Training Period (`step <= 504`)
- Transactions: 1,160,709
- Cold-Start Edges: 1,159,599 (99.90%)
- Cold-Starts with Dest Context: 484,423 (41.78%)

### Testing Period (`step > 504`)
- Transactions: 58,969
- Cold-Start Edges: 58,871 (99.83%)
- Cold-Starts with Dest Context: 23,412 (39.77%)

### Boundary Check
- **Test Edges utilizing pre-existing history**: 23,438 out of 58,969
- **Status**: Destination context successfully crosses the step 504 boundary, allowing test-period transactions to leverage training-period structural history. All statistics were explicitly evaluated without leaking future bounds.

## 6. Execution Details
- **Files Created**: 
  - `results/graph/phase5_cold_start_statistics.json`
  - `reports/phase5_cold_start_graph_analysis.md`
- **Memory Extracted (`paysim_dev_graph.npz`)**: 147.78 MB
- **Runtime**: 1.83 seconds
- **Limitations**: This is a strictly descriptive analysis. It maps structural availability but does not assess predictive weight directly (Phase 4 covered RF baseline prediction weight, and Phase 6 will cover GNN prediction).
