# Phase 5: Graph Architecture & Design Audit

## 1. Audit Summary
I conducted an architectural audit of the FraudDNA pipeline up to Phase 4.
**Inspected Components:**
- **Data Preprocessing (`src/data/preprocess.py`)**: Validated that `step` is used for temporal sorting, duplicates are handled, and `isFlaggedFraud` is strictly dropped to prevent leakage.
- **Feature Pipeline (`src/features/pipeline.py`)**: Confirmed standard transaction features (`amount`, `old/new balances`, `type` dummies).
- **FraudDNA Core (`src/fraud_dna/fingerprint.py`, `destination_fingerprint.py`, `build_baseline_features.py`)**: Audited the cumulative, leakage-free historical baseline construction. Discovered that Originator history is virtually non-existent (~0.09%), while Destination history is abundant (~41.6%).
- **Data Schema (`docs/data_dictionary.md`)**: Confirmed availability of `nameOrig` and `nameDest` identifiers which act as natural network connections.

## 2. Graph Definition
Based on the highly relational nature of the PaySim dataset and the need to connect cold-start originators to history-rich destinations, the optimal structure is a **Directed Multigraph**.

**Topology:** `Originator Node` —[ `Transaction Edge` ]—> `Destination Node`

Since PaySim entities can act as both originators and destinations (though rarely), the graph will merge `nameOrig` and `nameDest` into a single universal `Entity` node set. 

## 3. Schemas

### Node Schema
- **Type**: `Entity` (represents a unique PaySim user/merchant identifier).
- **ID**: Global integer mapping of the raw string IDs (e.g., `C123456...` $\rightarrow$ `0`).

### Edge Schema
- **Type**: `Transaction`
- **Direction**: Directed from source `Entity` to target `Entity`.
- **Temporal property**: Each edge contains a `step` attribute defining exactly when the transaction occurred.

### Feature Schema
Because FraudDNA's behavioral metrics (like mutation score) are calculated *relative to the exact time of the transaction*, they are intrinsically transaction-level temporal features, not static node features.

**Node Features:**
- No static node features are available in PaySim. We will initialize nodes with a constant vector (e.g., `[1.0]`) or a simple learnable embedding. The GNN will rely entirely on aggregating the rich edge features.

**Edge Features (`edge_attr`):**
1. **Raw Transaction:** `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `type_*` (dummies).
2. **Behavioral Context:** `origin_history_count`, `destination_history_count`.
3. **FraudDNA Mutations:** `origin_mutation`, `destination_mutation`.

**Edge Label (`y`):**
- `isFraud` (0 or 1). The prediction task is **Edge Classification**.

## 4. Addressing Core Challenges

### Temporal Strategy & Leakage Prevention
GNNs are highly susceptible to future data leakage. If we perform message passing on a static, fully-connected graph, a transaction at `step 10` might aggregate information from a transaction at `step 700`.

**Leakage Prevention Strategy:**
We will use a **Temporal Graph Network (TGN)** approach, or at minimum, a **Timestamp-Masked Message Passing** strategy. 
- When predicting the label for a transaction edge $e_{ij}$ at $t$, the GNN is only permitted to aggregate messages across edges where $step < t$.
- Future edges are strictly masked out of the adjacency matrix for that specific forward pass.

### Cold-Start Handling
In Phase 4, we proved Originators are 99% cold-starts, meaning they have no self-history.
In the graph representation, a cold-start originator is simply a node with an *in-degree and out-degree of 0* just prior to the transaction. 
During message passing, the GNN will flow information backward from the target `Destination` node (which has high degree/rich history) to the `Originator`. This inherently solves the cold-start problem by providing the transaction with structural relational context, mirroring our successful "Adaptive Context" from Phase 4, but doing so organically via graph topology.

## 5. Proposed Train/Test Construction
We will strictly maintain the chronological split proven in Phase 4:
- **Training Graph**: Constructed exclusively from edges where `step <= 504`. The model learns structural patterns and edge classification here.
- **Testing Graph**: Constructed using edges where `step > 504`. 
  - *Crucial detail*: When evaluating a test edge at $t_{test}$, the GNN *can* traverse historical edges from the training set, as well as earlier edges from the testing set ($t_{train} < t_{earlier\_test} < t_{test}$). This mimics a true online streaming environment.

## 6. Phase 6 Interface Requirements
To implement this design in Phase 6, the data pipeline must be extended to output standard Graph ML tensors (compatible with `PyTorch Geometric` or `DGL`):
1. `edge_index`: Tensor of shape `[2, num_transactions]` containing source/target node IDs.
2. `edge_attr`: Tensor of shape `[num_transactions, num_features]`.
3. `edge_time`: Tensor of shape `[num_transactions]` containing the `step`.
4. `y`: Tensor of shape `[num_transactions]` containing `isFraud`.
5. `train_mask` / `test_mask`: Boolean tensors defining the chronological split.
