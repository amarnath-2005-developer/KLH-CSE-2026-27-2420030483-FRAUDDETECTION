# Phase 6.1: GNN Environment and Dataset Smoke Test

## 1. Environment Audit
- **Python**: 3.13.5
- **PyTorch**: 2.8.0+cpu
- **NumPy**: 2.2.6
- **SciPy**: 1.16.3
- **PyTorch Geometric (PyG)**: Not installed (MISSING DEPENDENCY)

### Hardware
- **CPU**: Intel64 Family 6 Model 189 Stepping 1, GenuineIntel
- **Architecture**: AMD64
- **CUDA Available**: False
- **GPU**: None (CPU Mode only)

## 2. Dataset Loading
- **Node Count**: 1,929,718
- **Edge Count**: 1,219,678
- **Edge Feature Count**: 17
- **edge_index**: shape=[2, 1219678], dtype=torch.int64
- **edge_attr**: shape=[1219678, 17], dtype=torch.float32
- **edge_time**: shape=[1219678], dtype=torch.int64
- **y (Target)**: shape=[1219678], dtype=torch.int64

## 3. Node Feature Strategy
The validated Phase 5 graph does not contain intrinsic static node features. For this smoke test (and future GNN experiments), we must adopt a minimal node representation strategy. Allocating a unique learnable embedding for all 1,929,718 nodes would consume enormous memory and likely overfit given 99.9% cold-start sparsity.
**Strategy Chosen**: A constant scalar feature `torch.ones((num_nodes, 1))` for all nodes. This acts purely as an anonymous structural anchor, allowing the GNN to aggregate the rich *edge features* (which contain all behavioral/mutation context). *Note: This constant feature contains NO behavioral information.*

## 4. Memory Safety & Estimates
- **Node Features (`x`)**: 7.36 MB
- **Edge Index**: 18.61 MB
- **Edge Attributes**: 79.10 MB
- **Total Base Tensors**: 105.07 MB
*(Graph easily fits into system memory. No mini-batching strictly necessary for RAM, but mini-batching may be required if moving to a low-VRAM GPU).*

## 5. PyG Compatibility & Minimal Model Test
**CRITICAL ISSUE**: PyTorch Geometric is not installed in the current environment.
As per instructions, dependencies were not automatically upgraded.

**Dummy Forward-Pass (Pure PyTorch Fallback)**:
To verify memory safety and message passing feasibility without PyG, a minimal linear layer was applied to the edge attributes to predict the target label.
- **Smoke Test Status**: PASS (Pure PyTorch fallback). Model output shape: [1219678]. Loss calculation successful.

## 6. Discovered Issues & Recommendations for Phase 6.2
- **ISSUE 1**: `torch_geometric` is missing from the Python environment.
- **RECOMMENDATION**: Install PyTorch Geometric before beginning Phase 6.2 model training.
- **RECOMMENDATION**: Utilize a constant node feature `x = torch.ones(N, 1)` to drastically minimize memory footprint while allowing the GNN to learn structural embeddings strictly through the rich transaction `edge_attr` context.