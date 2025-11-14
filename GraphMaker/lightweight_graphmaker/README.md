# Lightweight GraphMaker

A simplified, self-contained implementation of graph diffusion models for studying **memorization vs generalization** in graph generation.

## Overview

This implementation focuses on:
- ✅ **Bias-free models** - All parameters have no bias terms
- ✅ **Edge generation only** - Focuses on graph structure (not node features)
- ✅ **Multi-graph training** - Trains on datasets of 500 graphs
- ✅ **PyTorch Geometric** - Uses PyG instead of DGL
- ✅ **Discrete diffusion** - Marginal-based corruption for categorical data
- ✅ **WL kernel evaluation** - Weisfeiler-Lehman similarity for comparing graph distributions

## Research Question

**Do diffusion models memorize training data or learn the underlying distribution?**

### Hypothesis

- **Small graphs (n=20, 50)**: Models **memorize** their specific training sets
  - `WLSim(Gen1, S1)` >> `WLSim(Gen1, Gen2)`
  - `WLSim(Gen2, S2)` >> `WLSim(Gen1, Gen2)`

- **Large graphs (n=100, 500)**: Models **generalize** to the same distribution
  - `WLSim(Gen1, Gen2)` >> `WLSim(Gen1, S1)`
  - `WLSim(Gen1, Gen2)` >> `WLSim(Gen2, S2)`

### Experimental Setup

1. Train **DF1** on **S1** (high homophily, 500 graphs)
2. Train **DF2** on **S2** (low homophily, 500 graphs)
3. Generate 100 graphs from each model
4. Compute WL kernel similarities
5. Repeat for n = 20, 50, 100, 500

## Installation

```bash
# Already in GraphMaker environment
conda activate pygeo310

# Install additional dependency for WL kernel
pip install grakel
```

## Project Structure

```
lightweight_graphmaker/
├── dataset.py          # Multi-graph dataset loader
├── diffusion.py        # Discrete diffusion (forward & reverse)
├── model.py           # Bias-free GNN denoiser
├── train.py           # Training script for DF1/DF2
├── sample.py          # Generation from trained models
├── wl_kernel.py       # Weisfeiler-Lehman similarity
└── experiment.py      # Full experiment pipeline
```

## Quick Start

### 1. Test Individual Components

```bash
# Test dataset loading
python lightweight_graphmaker/dataset.py

# Test diffusion module
python lightweight_graphmaker/diffusion.py

# Test bias-free model
python lightweight_graphmaker/model.py

# Test WL kernel
python lightweight_graphmaker/wl_kernel.py
```

### 2. Train a Single Model

```bash
# Train DF1 on S1 for node_size=20
python lightweight_graphmaker/train.py \
    --node_size 20 \
    --split S1 \
    --model_name DF1 \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 1e-3

# Train DF2 on S2 for node_size=20
python lightweight_graphmaker/train.py \
    --node_size 20 \
    --split S2 \
    --model_name DF2 \
    --num_epochs 100
```

### 3. Generate Graphs

```bash
# Generate from trained model
python lightweight_graphmaker/sample.py \
    --checkpoint checkpoints/DF1_n20_best.pt \
    --num_samples 100 \
    --output generated_DF1_n20.pt
```

### 4. Run Full Experiment (Single Node Size)

```bash
# Run complete pipeline for node_size=20
python lightweight_graphmaker/experiment.py \
    --single_node_size 20 \
    --num_epochs 100 \
    --num_generated 100
```

### 5. Run Full Study (All Node Sizes)

```bash
# Run experiments for multiple node sizes
python lightweight_graphmaker/experiment.py \
    --node_sizes 20 50 100 \
    --num_generated 100 \
    --device cuda
```

## Model Architecture

### Discrete Diffusion Process

**Forward Process (Corruption)**:
```
E_0 → E_1 → ... → E_T
```
- Gradually corrupts edges toward marginal distribution
- Uses cosine noise schedule for α_t
- Transition: `Q_t = α_t * I + (1 - α_t) * Marginal`

**Reverse Process (Denoising)**:
```
E_T → E_{T-1} → ... → E_0
```
- Starts from noise
- Iteratively predicts E_0 from E_t
- Samples from posterior q(E_{t-1} | E_t, pred_E_0)

### Bias-Free GNN Denoiser

```python
Input: (edge_index_t, Y, t)
  ↓
Label Embedding (no bias)
  ↓
Time Embedding (MLP, no bias)
  ↓
Message Passing Layers (3 layers, no bias)
  ↓
Edge Predictor (for all node pairs, no bias)
  ↓
Output: Edge logits [num_possible_edges, 2]
```

**Key Features**:
- ✅ Zero bias parameters (verified)
- ✅ Residual connections
- ✅ Time conditioning
- ✅ Label conditioning
- ✅ Predicts all possible edges simultaneously

## Dataset Format

Your datasets in `data/node_xx/`:

```python
# Each .pt file contains:
List[(graph, metadata)]

# Where:
graph = Data(
    x=[num_nodes, 11],           # Node features
    edge_index=[2, num_edges],   # Edge connectivity
    y=[num_nodes],               # Node labels
    num_nodes=num_nodes
)

metadata = {
    'seed': 'Cora',
    'target_label_hom': 0.90,
    'realised_label_hom': 0.84,
    'avg_degree': 2.5,
    'num_edges': 25,
    'stats': [15 graph statistics]
}
```

## Results

Experiments save to `experiments/nXX_TIMESTAMP/`:

```
experiments/n20_20251114_123456/
├── config.json                  # Experiment configuration
├── results.json                 # WL similarity scores
├── gen1_n20.pt                 # Generated graphs from DF1
├── gen2_n20.pt                 # Generated graphs from DF2
└── checkpoints/
    ├── DF1_n20_best.pt         # Best DF1 checkpoint
    ├── DF1_n20_loss.png        # DF1 training curve
    ├── DF2_n20_best.pt         # Best DF2 checkpoint
    └── DF2_n20_loss.png        # DF2 training curve
```

### Results Interpretation

```json
{
  "WLSim_Gen1_S1": 0.85,      // DF1 memorization
  "WLSim_Gen2_S2": 0.82,      // DF2 memorization
  "WLSim_Gen1_Gen2": 0.65,    // Cross-model similarity
  "gen_vs_mem_ratio": 0.78    // Ratio < 1 → Memorization
}
```

- **Ratio > 1.2**: Models **generalize** (learn same distribution)
- **Ratio < 0.8**: Models **memorize** (distinct training sets)
- **0.8 ≤ Ratio ≤ 1.2**: Intermediate regime

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | GNN hidden dimension |
| `num_layers` | 3 | Number of message passing layers |
| `num_timesteps` | 100 | Diffusion timesteps T |
| `batch_size` | 16 | Training batch size |
| `lr` | 1e-3 | Learning rate |
| `num_epochs` | 100 (n=20/50), 200 (n=100), 300 (n=500) | Training epochs |

## Key Differences from Original GraphMaker

| Aspect | Original | Lightweight |
|--------|----------|-------------|
| **Framework** | DGL | PyTorch Geometric |
| **Data** | Single graph | Multi-graph (500 per split) |
| **Features** | Node attributes + edges | Edges only (structure) |
| **Bias** | Has bias | **Bias-free** |
| **Focus** | Generation quality | **Memorization vs generalization** |
| **Evaluation** | MMD, discriminator | **WL kernel similarity** |
| **Complexity** | ~1400 lines | ~800 lines |

## Citation

Inspired by:
```bibtex
@article{li2024graphmaker,
    title={GraphMaker: Can Diffusion Models Generate Large Attributed Graphs?},
    author={Mufei Li and Eleonora Kreačić and Vamsi K. Potluru and Pan Li},
    journal={Transactions on Machine Learning Research},
    year={2024}
}
```

## Notes

- Models are **deterministic** given seed (for reproducibility)
- WL kernel computation can be slow for large graphs
- GPU recommended for n=100, 500
- Expected training time: ~10min (n=20), ~30min (n=50), ~2hrs (n=100)
