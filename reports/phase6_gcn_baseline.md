# Phase 6.3: First GNN Baseline (Edge-Aware GCN)

## 1. Overview
This phase evaluates an edge-aware GCN baseline against the Phase 4 Machine Learning baselines. Due to the online-transactional nature of PaySim and the necessity of preventing future structural leakage, a mathematically safe "Frozen Structural State" evaluation strategy was implemented.

## 2. Experimental Setup
- **Training Subset**: Transactions where `step <= 504`. (Message-passing was tightly restricted to these edges to build the structural node history).
- **Test Subset**: Transactions where `step > 504`. (Evaluated strictly via the frozen step-504 node representations + native test `edge_attr`).
- **Graph Demographics**: 1,929,718 nodes, 1,219,678 edges.

### Architectures
1. **Model A (GCN + Edge Features)**: 
   - **Node Features**: `x = torch.ones(num_nodes, 1)` (Minimal constant scalar to prevent parameter explosion).
   - **GNN Backbone**: 2 layers of `GCNConv` (Hidden dimension: 16).
   - **Edge Classifier**: MLP projecting `[src_emb (16) || dst_emb (16) || edge_attr (17)]` → `Linear(32)` → `ReLU` → `Linear(1)`.
   - **Total Parameters**: 1,937
   - **Training Time**: ~74.9 seconds
   
2. **Model B (Edge-Only MLP Ablation)**:
   - **Architecture**: Identical classifier architecture operating *only* on the 17 `edge_attr` features, skipping `GCNConv` message passing entirely.
   - **Total Parameters**: 609
   - **Training Time**: ~7.7 seconds

## 3. Results Overview (Testing Period)
The testing set consists of 58,969 transactions containing 2,592 fraud labels.

| Metric | Phase 4 RF (Behavioral Baseline) | Model A (GCN + Edge) | Model B (Edge-Only MLP) |
|--------|----------------------------------|----------------------|-------------------------|
| **PR-AUC** | 0.6121 | 0.8088 | **0.8099** |
| **ROC-AUC**| N/A | 0.9688 | 0.9683 |
| **Precision**| N/A | 0.1954 | 0.2565 |
| **Recall** | N/A | 0.9375 | 0.8931 |
| **F1-Score**| N/A | 0.3234 | 0.3985 |

*(Note: Random Forest Phase 4 Precision/Recall/F1 data was not loaded for side-by-side comparison here, but PR-AUC serves as the primary evaluation anchor.)*

## 4. Cold-Start Analysis
99.9% of originators in the test set had absolutely zero historical training connections (`origin_history_count == 0`). 
When filtering test predictions to explicitly isolate these **cold-start originators**:

| Cold-Start Subset | GCN (Model A) PR-AUC | MLP (Model B) PR-AUC |
|-------------------|----------------------|----------------------|
| All Cold-Starts | 0.8088 | 0.8099 |
| Dest History **Available** | 0.5780 | - |
| Dest History **Unavailable** | 0.8769 | - |

*(Note: The higher PR-AUC on unavailable destination history reflects PaySim's structural skew where specific types of immediate cash-outs/transfers are highly fraudulent despite zero destination footprint).*

## 5. Critical Takeaways and Ablation Findings
1. **Massive Overall Leap**: Transitioning from Phase 4's Random Forest (PR-AUC 0.6121) to neural network representations resulted in a monumental leap to ~0.809 PR-AUC. This suggests Neural Networks handle the normalized cumulative FraudDNA features exceptionally well.
2. **Structural Impotence**: The GCN (Model A) actually performed *fractionally worse* (or equivalently) to the Edge-Only MLP (Model B). 
   - **Conclusion**: In a highly sparse graph where 99.9% of the interacting entities are cold-starts and nodes carry no intrinsic static features (`x = 1.0`), standard 2-hop graph convolutions add **zero predictive value** beyond what is already cleanly summarized by the raw 17 edge attributes (which explicitly contain the historical transaction counts and mutation scores!).
3. **Efficiency**: The pure tabular MLP trained exactly 10x faster and required 3x fewer parameters while delivering slightly superior F1 and Precision metrics.

## 6. Limitations
- The "Frozen Structural State" evaluation guarantees zero future leakage but prevents the GCN from "updating" its node representations mid-test-stream. However, given that the MLP perfectly matched the GCN, it is clear that the edge attributes (which *are* temporally updated per transaction via Phase 4) completely dominate the predictive signal. 
