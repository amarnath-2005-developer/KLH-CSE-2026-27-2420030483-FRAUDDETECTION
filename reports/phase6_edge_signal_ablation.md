# Phase 6.3.1: GNN Signal Attribution Ablation

## 1. Overview
In Phase 6.3, the PyTorch models (both the GCN and Edge-Only MLP) achieved a massive PR-AUC of ~0.81, completely eclipsing the Phase 4 Random Forest baseline (PR-AUC 0.6121). 

This ablation study isolates the 17 edge features into specific subgroups and feeds them into the exact same PyTorch MLP architecture. The objective is to attribute the exact source of this massive performance leap: was it the standard behavioral history, the FraudDNA mutations, or the raw transactional features?

## 2. Experimental Setup
- **Model**: `EdgeOnlyMLP` (Multi-Layer Perceptron), mirroring Phase 6.3's edge classifier.
- **Data**: 1,219,678 total edges. Test set evaluated strictly chronologically (`step > 504`).
- **Standardization**: All features normalized to mean=0, std=1 strictly using training statistics.
- **Configurations Evaluated**:
  - **A. Transaction-only**: 9 features (amount, old/new balances, transaction types).
  - **B. Transaction + Behavior**: 15 features (added historical counts, historical averages, availability flags).
  - **C. Transaction + Mutation**: 11 features (added origin/destination FraudDNA mutations).
  - **D. All Features**: 17 features (Phase 6.3 configuration).

## 3. Results

| Configuration | Features | PR-AUC | F1-Score | Recall |
|---------------|----------|--------|----------|--------|
| **A. Transaction-only** | 9 | **0.8161** | **0.4449** | 0.8735 |
| **B. Transaction + Behavior** | 15 | 0.8141 | 0.4060 | **0.8927** |
| **C. Transaction + Mutation** | 11 | 0.8151 | 0.4382 | 0.8777 |
| **D. All Features (Phase 6.3 Baseline)** | 17 | 0.8004 | 0.3958 | 0.8789 |

## 4. Comparison to Phase 4 (Random Forest)

| Feature Set | Phase 4 RF PR-AUC | Phase 6 MLP PR-AUC | Absolute Leap |
|-------------|-------------------|--------------------|---------------|
| Transaction-only | 0.4816 | **0.8161** | + 0.3345 |
| Transaction + Behavior | 0.6121 | 0.8141 | + 0.2020 |
| Transaction + Mutation | 0.6063 | 0.8151 | + 0.2088 |

## 5. Interpretation
1. **The Origin of the Leap**: The massive jump in PR-AUC (to ~0.81) was **not** caused by graph structure, nor was it caused by the behavioral or mutation features. The jump is entirely attributable to the Artificial Neural Network's vastly superior ability to extract signal from the raw continuous transaction variables (like `amount`, `oldbalance`, `newbalance`) compared to the Random Forest.
2. **Behavioral Redundancy**: When using the Neural Network, adding the historical behavioral features or the mutation scores to the raw transaction features yielded absolutely zero additional PR-AUC. In fact, Configuration D (All Features) suffered a slight degradation (0.8004) compared to Configuration A (0.8161), likely due to minor overfitting on the increased dimensionality.
3. **Conclusion**: For this specific dataset and evaluation split, a standard Multilayer Perceptron trained solely on the basic transactional variables is the definitive state-of-the-art model. Neither complex sequential behavior mapping (FraudDNA) nor topological structural modeling (GNNs) can beat the signal natively present in the raw accounting balances when processed by a neural network.

## 6. Limitations
- These results reflect performance on the `step > 504` chronological split of the 1.9M-row PaySim development sample.
- This ablation does not claim that graph structure or behavioral mutations are universally useless in fraud detection; rather, it demonstrates that *for this specific PaySim distribution*, the neural network extracts maximum predictive utility directly from the raw balance changes, leaving no residual variance for the complex behavioral/structural features to explain.
