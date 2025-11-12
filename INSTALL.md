# Neural Graph Generator - Environment Setup

This project uses **PyTorch**, **PyTorch Geometric**, and various scientific computing libraries for graph neural network research.

## Quick Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate pygeo310
```

### Option 2: pip + virtualenv

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## GPU Support

If you have an NVIDIA GPU with CUDA support:

### For CUDA 11.8:

```bash
# Edit environment.yml: remove 'cpuonly' and add 'pytorch-cuda=11.8'
conda env create -f environment.yml
```

Or with pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt
```

### For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
pip install -r requirements.txt
```

## Verify Installation

```python
import torch
import torch_geometric
import networkx as nx

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NetworkX version: {nx.__version__}")
```

## Core Dependencies

- **Python**: 3.10
- **PyTorch**: 2.0.1+ (Deep learning framework)
- **PyTorch Geometric**: 2.3.0+ (Graph neural networks)
- **NetworkX**: 2.8.0+ (Graph manipulation and visualization)
- **NumPy**: 1.23.0+ (Numerical computing)
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

## Project Structure

```
.
├── autoencoder.py          # Variational Graph Autoencoder (VGAE)
├── denoise_model.py        # Latent Diffusion Model (LDM)
├── main_nodesize.py        # Main training script
├── evaluate_saved_models.py # Evaluation without retraining
├── analyze_latent_jacobian.py # Jacobian analysis for memorization
├── utils.py                # Utility functions
├── data/                   # Dataset storage
└── outputs/                # Training outputs and checkpoints
```

## Running Experiments

### Train Models
```bash
conda activate pygeo310
python main_nodesize.py --node-sizes 20 50 100 --N-train 500 --wl-iter 5
```

### Evaluate Saved Models
```bash
python evaluate_saved_models.py \
    --checkpoint-dir outputs/nodesize_study/WL_iter=5 \
    --node-sizes 20 50 100 \
    --N 500
```

### Analyze Jacobian
```bash
python analyze_latent_jacobian.py \
    --node-sizes 20 100 500 \
    --exp-dir outputs/nodesize_study/WL_iter=5 \
    --N-train 500
```

## Troubleshooting

### PyTorch Geometric Installation Issues

If you encounter issues with PyG extensions:

```bash
# Try installing from source
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.0.0+cpu.html
```

### CUDA Version Mismatch

Check your CUDA version:
```bash
nvidia-smi  # Check driver CUDA version
nvcc --version  # Check toolkit CUDA version
```

Then install matching PyTorch version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/documentation/)
