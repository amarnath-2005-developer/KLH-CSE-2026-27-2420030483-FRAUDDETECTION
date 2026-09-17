# Phase 5.4: Destination Behavioral Context Validation

## 1. Overview
This report validates the successful integration of Destination Behavioral Context into the PyTorch Geometric compatible transaction graph. As discovered in Phase 4, destination-side behavioral information is substantially more useful than originator self-history in PaySim. This context is now physically represented on the edges of the temporal graph, adhering strictly to the existing FraudDNA implementation.

## 2. Included Destination Features & Sources
Per the strict instruction to only include features *already correctly implemented* by the existing Phase 4 pipeline, the following features have been successfully merged into the graph `edge_attr`:

| Feature | Source | Historical Availability |
|---------|--------|--------------------------|
| `destination_history_count` | Phase 4 Parquet | Historical (Shifted Cumcount) |
| `destination_history_avg_amount` | Phase 4 Parquet | Historical (Shifted Avg) |
| `destination_mutation` | Phase 4 Parquet | Hist/Curr Relative |
| `destination_history_available` | Phase 4 Parquet | Historical Flag |

*(Note: Features like velocity, unique originators, and type distribution were not natively exported by the `fast_cumulative_features` implementation in Phase 4. Adhering to the rule "Do not redesign the existing FraudDNA feature pipeline", they were safely omitted.)*

## 3. Context Coverage Statistics
- **Total Transactions (Edges)**: 1,219,678
- **Originator Cold-Start Coverage**: 99.90% (1,218,470 edges have `origin_history_count == 0`).
- **Destination History Coverage**: ~41.6% of all transactions successfully map to a pre-existing destination history baseline.
- **Destination Mutation Coverage**: ~41.6% (Missing histories correctly retain `mutation = -1` and are distinctly mapped via the `destination_history_available` flag, avoiding false zeros).

## 4. Strict Validations Performed

**A. First Destination Transaction**
- **Verified**: A new destination correctly shows `destination_history_count = 0`, `destination_history_available = 0`, and `destination_mutation = -1`. The current transaction does not leak into its own baseline.

**B. Repeated Destination Transaction**
- **Verified**: The previous transaction forms the baseline. For example, manual tracing of `C1286084959` confirmed the history count seamlessly iterates (0, 1, 2, 3...) exactly prior to the respective transaction.

**C. Multiple Transactions at the Same Step**
- **Verified**: Preserved raw-log sequential ordering. When multiple transactions hit the same destination in the same hour, they do not aggregate each other; the first strictly acts as history for the second.

**D. Step 504/505 Boundary**
- **Verified**: Train and test sets remain strictly disjoint (`step <= 504` vs `step > 504`). Historical destination information flows cleanly across the boundary without peeking at the future.

**E. Cold-Start Originator Connection**
- **Verified**: The structural graph design means that 99.9% of originators have no history, but when their transaction edge connects to a destination node, that edge successfully carries the destination's rich ~41.6% history.

**F. Label Leakage**
- **Verified**: `isFraud` and `isFlaggedFraud` do not exist in the destination history metrics or the edge attributes matrix.

## 5. System Metrics
- **Files Modified**: `src/graph/builder.py` (added `destination_history_avg_amount` and `origin_history_avg_amount` to feature schema).
- **Files Created**: `reports/phase5_destination_context_validation.md`
- **Graph Builder Runtime**: 23.94 seconds
- **Output Graph Memory (`.npz`)**: 126.84 MB
- **Total Edge Features**: 17

## 6. Limitations
- Destination context acts as a powerful proxy for behavioral history, but it does not technically solve originator cold-start fraud detection on its own. It merely provides relational anchor context for the transaction. 
- Features like exact incoming velocity (time gaps) and unique originator counts were not supported by the existing pipeline and are thus absent from this graph version.
