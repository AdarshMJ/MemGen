"""
Graph Generation: Memorization to Generalization Transition Study

This script trains models on different dataset sizes to study the transition from 
memorization to generalization as a function of dataset size vs task complexity.

Experiment:
- Train models on subsets of increasing size N
- Study when models transition from memorization (small N) to generalization (large N)
- Track complexity/dataset_size ratio to understand transition threshold

Dataset: labelhomophily0.5_10nodes_graphs.pkl
- Total: 2100 graphs (10 nodes each, label homophily = 0.5)
- Split: 1000 graphs for S1, 1000 graphs for S2, 100 graphs for conditioning/testing

Note: Using label homophily instead of feature homophily, but since the generator 
is not conditioned on this property, it should not affect the results.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import pandas as pd

from torch_geometric.loader import DataLoader
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import (linear_beta_schedule, construct_nx_from_adj, 
                   eval_autoencoder, gen_stats)
from grakel import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

# Configuration - Training 10 models with varying dataset sizes
# Dataset: 2100 graphs total (1000 for S1, 1000 for S2, 100 for conditioning)
N_VALUES = [10, 20, 50, 100, 200, 500, 1000]  # 7 different training set sizes (up to 1000)
TEST_SET_SIZE = 100  # Number of conditioning graphs held out from both models
SPLIT_SEED = 42  # Reproducible shuffle before creating S1/S2/test splits
NUM_SAMPLES_PER_CONDITION = 5  # Number of samples to draw per model per conditioning vector
KS_SIGNIFICANCE_THRESHOLD = 0.05  # Significance level for KS tests comparing S1/S2 statistics
BETA_KL_WEIGHT = 0.05
SMALL_DATASET_THRESHOLD = 50
SMALL_DATASET_KL_WEIGHT = 0.01
SMALL_DATASET_DROPOUT = 0.1

# Training hyperparameters - KEPT CONSTANT ACROSS ALL N VALUES
# This ensures no hidden confounders - only training set size varies
EPOCHS_AUTOENCODER = 100  # Increased from 100 for better convergence
EPOCHS_DENOISER = 100  # Increased from 50 for better convergence
EARLY_STOPPING_PATIENCE = 50  # Stop if no improvement for 30 epochs
BATCH_SIZE = 32  # Same batch size for all experiments
LEARNING_RATE = 0.0001  # Same learning rate for all experiments
GRAD_CLIP = 1.0  # Gradient clipping to prevent exploding gradients
LATENT_DIM = 32  # Same latent dimension for all experiments
HIDDEN_DIM_ENCODER = 32  # Same encoder architecture for all experiments
HIDDEN_DIM_DECODER = 64  # Same decoder architecture for all experiments
HIDDEN_DIM_DENOISE = 512  # Same denoiser architecture for all experiments
N_MAX_NODES = 500  # Maximum nodes (capped at 500 for computational feasibility)
N_PROPERTIES = 15  # Conditioning properties include homophily measurements
TIMESTEPS = 500  # Same diffusion timesteps for all experiments

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_dataset(data_path='data/labelhomophily0.5_10nodes_graphs.pkl'):
    """Load the graph dataset."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} graphs")

    # Inject measured homophily values from log if available so conditioning stays accurate
    log_path = Path(data_path).with_name(Path(data_path).name.replace('_graphs.pkl', '_log.csv'))
    homophily_log = None
    if log_path.exists():
        try:
            homophily_log = pd.read_csv(log_path).set_index('graph_idx')
            print(f"Loaded homophily log from {log_path}")
        except Exception as exc:
            print(f"Warning: failed to load homophily log ({exc}).")

    for idx, data in enumerate(data_list):
        if homophily_log is not None:
            log_idx = int(getattr(data, 'graph_idx', idx))
            if log_idx in homophily_log.index:
                row = homophily_log.loc[log_idx]
                if hasattr(data, 'feature_homophily'):
                    data.feature_homophily = torch.tensor(float(row['actual_feature_hom']))
                if hasattr(data, 'stats'):
                    stats = data.stats
                    if isinstance(stats, torch.Tensor):
                        if stats.dim() == 1:
                            stats = stats.unsqueeze(0)
                        if stats.size(-1) < 18:
                            stats = torch.nn.functional.pad(stats, (0, 18 - stats.size(-1)))
                        data.stats = stats
                        data.stats[0, -3:] = torch.tensor([
                            float(row['actual_label_hom']),
                            float(row['actual_structural_hom']),
                            float(row['actual_feature_hom'])
                        ])
                if not hasattr(data, 'graph_idx'):
                    data.graph_idx = log_idx
    
    # Basic info
    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Graph properties: {sample.x.shape[1]}D features, {sample.x.shape[0]} nodes")
        if hasattr(sample, 'feature_homophily'):
            print(f"Feature homophily: {sample.feature_homophily:.2f}")
    
    return data_list


def shuffle_and_split_dataset(data_list, test_size=TEST_SET_SIZE, seed=SPLIT_SEED, stats_cache_path=None):
    """Shuffle dataset and create reusable pools for S1, S2, and held-out test graphs."""
    total_graphs = len(data_list)
    if total_graphs < test_size + 2 * max(N_VALUES):
        raise ValueError(
            f"Dataset too small for requested configuration: need at least "
            f"{test_size + 2 * max(N_VALUES)} graphs, found {total_graphs}."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(total_graphs)
    rng.shuffle(indices)

    shuffled_data = [data_list[idx] for idx in indices]

    test_graphs = shuffled_data[-test_size:]
    test_indices = indices[-test_size:]

    train_pool = shuffled_data[:-test_size]
    train_indices = indices[:-test_size]

    half = len(train_pool) // 2
    if half < max(N_VALUES):
        raise ValueError(
            f"Not enough data in each training split after shuffling. "
            f"Available per split: {half}, required: {max(N_VALUES)}"
        )

    S1_pool = train_pool[:half]
    S1_indices = train_indices[:half]
    S2_pool = train_pool[half:]
    S2_indices = train_indices[half:]

    test_stats_cache = None
    if stats_cache_path is not None and stats_cache_path.exists():
        try:
            cache_payload = torch.load(stats_cache_path, map_location='cpu')
            cache_props = cache_payload.get('n_properties')
            if cache_payload.get('test_size') == test_size and cache_payload.get('seed') == seed and cache_props == N_PROPERTIES:
                cached_indices = np.asarray(cache_payload.get('test_indices'))
                if cached_indices is not None and np.array_equal(cached_indices, test_indices):
                    cached_stats = cache_payload.get('stats')
                    if isinstance(cached_stats, torch.Tensor):
                        test_stats_cache = cached_stats.float()
                    elif cached_stats is not None:
                        test_stats_cache = torch.tensor(cached_stats, dtype=torch.float32)
                    if test_stats_cache is not None:
                        print(f"Loaded cached conditioning stats from {stats_cache_path}")
        except Exception as exc:
            print(f"Warning: failed to load conditioning cache ({exc}); recomputing.")

    if test_stats_cache is None:
        stats_matrix = stack_stats(test_graphs)
        if stats_matrix.size > 0:
            test_stats_cache = torch.from_numpy(stats_matrix).float()
        else:
            test_stats_cache = torch.empty((0, N_PROPERTIES), dtype=torch.float32)

        if stats_cache_path is not None:
            try:
                stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'seed': seed,
                    'test_size': test_size,
                    'test_indices': test_indices,
                    'stats': test_stats_cache.cpu(),
                    'n_properties': N_PROPERTIES
                }, stats_cache_path)
                print(f"Saved conditioning stats cache to {stats_cache_path}")
            except Exception as exc:
                print(f"Warning: failed to save conditioning cache ({exc}).")

    return {
        'S1_pool': S1_pool,
        'S1_indices': S1_indices,
        'S2_pool': S2_pool,
        'S2_indices': S2_indices,
        'test_graphs': test_graphs,
        'test_indices': test_indices,
        'test_stats_cache': test_stats_cache,
        'stats_cache_path': stats_cache_path,
        'permutation': indices,
        'seed': seed,
        'test_size': test_size
    }


def create_splits(split_config, N):
    """Create non-overlapping splits of size N from pre-shuffled pools."""
    S1_pool = split_config['S1_pool']
    S2_pool = split_config['S2_pool']
    S1_indices_pool = split_config['S1_indices']
    S2_indices_pool = split_config['S2_indices']
    test_stats_cache = split_config.get('test_stats_cache')

    if N > len(S1_pool) or N > len(S2_pool):
        raise ValueError(
            f"Requested N={N} exceeds available pool sizes: "
            f"S1={len(S1_pool)}, S2={len(S2_pool)}"
        )

    S1 = S1_pool[:N]
    S2 = S2_pool[:N]
    S1_indices = S1_indices_pool[:N]
    S2_indices = S2_indices_pool[:N]
    test_graphs = split_config['test_graphs']
    test_indices = split_config['test_indices']

    print(f"\nData splits for N={N} (seed={split_config['seed']}):")
    print(f"  S1: {len(S1)} graphs | original indices min={int(np.min(S1_indices))}, "
          f"max={int(np.max(S1_indices))}")
    print(f"  S2: {len(S2)} graphs | original indices min={int(np.min(S2_indices))}, "
          f"max={int(np.max(S2_indices))}")
    print(f"  Test set: {len(test_graphs)} graphs | original indices range="
        f" [{int(np.min(test_indices))}, {int(np.max(test_indices))}]")
    if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) == len(test_graphs):
      print("  Test conditioning stats pulled from cache")

    return S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices


def graph_stats_to_numpy(data):
    """Convert stored graph statistics to a numpy array limited to N_PROPERTIES."""
    if not hasattr(data, 'stats'):
        raise AttributeError("Graph data object is missing required 'stats' attribute.")

    stats = data.stats
    if isinstance(stats, torch.Tensor):
        stats_np = stats.detach().cpu().numpy()
    else:
        stats_np = np.asarray(stats)

    if stats_np.ndim == 1:
        stats_np = stats_np.reshape(1, -1)

    if stats_np.shape[1] < N_PROPERTIES:
        padding = N_PROPERTIES - stats_np.shape[1]
        stats_np = np.pad(stats_np, ((0, 0), (0, padding)), constant_values=np.nan)

    return stats_np[:, :N_PROPERTIES]


def stack_stats(graph_list):
    """Stack per-graph statistics into a matrix of shape [len(graphs), N_PROPERTIES]."""
    if len(graph_list) == 0:
        return np.empty((0, N_PROPERTIES))

    stats_mats = []
    for data in graph_list:
        try:
            stats_mats.append(graph_stats_to_numpy(data))
        except Exception as exc:
            print(f"Warning: failed to extract stats for a graph ({exc}).")

    if len(stats_mats) == 0:
        return np.empty((0, N_PROPERTIES))

    return np.vstack(stats_mats)


def ks_2samp(sample1, sample2):
    """Two-sample Kolmogorov-Smirnov test (asymptotic p-value approximation)."""
    data1 = np.sort(np.asarray(sample1))
    data2 = np.sort(np.asarray(sample2))

    n1 = data1.size
    n2 = data2.size

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.abs(cdf1 - cdf2))

    en = np.sqrt(n1 * n2 / (n1 + n2))
    if en == 0:
        return d, np.nan

    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-8:
        return d, 1.0

    p_value = 0.0
    for j in range(1, 100):
        term = 2 * ((-1) ** (j - 1)) * np.exp(-2 * (lam ** 2) * (j ** 2))
        p_value += term
        if abs(term) < 1e-8:
            break

    p_value = float(max(0.0, min(1.0, p_value)))
    return d, p_value


def perform_distribution_checks(S1_subset, S2_subset, exp_dir, N):
    """Run KS tests on graph statistics to confirm S1/S2 parity."""
    stats_S1 = stack_stats(S1_subset)
    stats_S2 = stack_stats(S2_subset)

    output_path = exp_dir / f"ks_tests_N{N}.txt"
    results = []

    for feature_idx in range(N_PROPERTIES):
        col1 = stats_S1[:, feature_idx] if stats_S1.size else np.array([])
        col2 = stats_S2[:, feature_idx] if stats_S2.size else np.array([])

        col1 = col1[~np.isnan(col1)]
        col2 = col2[~np.isnan(col2)]

        if col1.size == 0 or col2.size == 0:
            d_val, p_val = np.nan, np.nan
        elif np.allclose(col1, col1[0]) and np.allclose(col2, col2[0]) and np.isclose(col1[0], col2[0]):
            d_val, p_val = 0.0, 1.0
        else:
            d_val, p_val = ks_2samp(col1, col2)

        results.append({
            'feature': feature_idx,
            'statistic': d_val,
            'p_value': p_val
        })

    significant = [r for r in results if not np.isnan(r['p_value']) and r['p_value'] < KS_SIGNIFICANCE_THRESHOLD]

    with open(output_path, 'w') as f:
        f.write(f"KS Tests for N={N}\n")
        f.write("Feature\tD-stat\tP-value\n")
        for row in results:
            f.write(f"{row['feature']}\t{row['statistic']:.4f}\t{row['p_value']}\n")
        f.write("\n")
        f.write(f"Significance threshold: {KS_SIGNIFICANCE_THRESHOLD}\n")
        f.write(f"Number of features failing (p < threshold): {len(significant)}\n")

    if significant:
        print(f"⚠️  KS test detected {len(significant)} / {N_PROPERTIES} features with distribution drift (see {output_path}).")
    else:
        print(f"KS tests passed for N={N}; all feature distributions aligned (see {output_path}).")

    return {
        'results': results,
        'num_failures': len(significant),
        'output_path': output_path
    }


def compute_within_model_similarity(graphs):
    """Compute pairwise WL similarities within a set of generated graphs."""
    if len(graphs) < 2:
        return []

    scores = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            scores.append(compute_wl_similarity(graphs[i], graphs[j]))
    return scores


def compute_mmd_rbf(latents_a, latents_b, bandwidths=None):
    """Compute squared MMD with an RBF kernel between two latent code sets."""
    if latents_a is None or latents_b is None:
        return np.nan

    if isinstance(latents_a, np.ndarray):
        latents_a = torch.from_numpy(latents_a)
    if isinstance(latents_b, np.ndarray):
        latents_b = torch.from_numpy(latents_b)

    if latents_a.numel() == 0 or latents_b.numel() == 0:
        return np.nan

    latents_a = latents_a.float()
    latents_b = latents_b.float()

    if latents_a.dim() == 1:
        latents_a = latents_a.unsqueeze(0)
    if latents_b.dim() == 1:
        latents_b = latents_b.unsqueeze(0)

    with torch.no_grad():
        combined = torch.cat([latents_a, latents_b], dim=0)
        dists = torch.cdist(combined, combined, p=2).pow(2)
        positive_dists = dists[dists > 0]
        if bandwidths is None:
            if positive_dists.numel() == 0:
                bandwidths = [1.0]
            else:
                median_sq = torch.median(positive_dists)
                gamma = 1.0 / (2.0 * median_sq)
                bandwidths = [gamma, gamma * 0.5, gamma * 2.0]

        if isinstance(bandwidths, (list, tuple)):
            gammas = [float(g) for g in bandwidths if g > 0]
        else:
            gammas = [float(bandwidths)]

        if not gammas:
            gammas = [1.0]

        d_xx = torch.cdist(latents_a, latents_a, p=2).pow(2)
        d_yy = torch.cdist(latents_b, latents_b, p=2).pow(2)
        d_xy = torch.cdist(latents_a, latents_b, p=2).pow(2)

        mmd_total = 0.0
        for gamma in gammas:
            k_xx = torch.exp(-gamma * d_xx)
            k_yy = torch.exp(-gamma * d_yy)
            k_xy = torch.exp(-gamma * d_xy)
            mmd_total += k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    return float(torch.clamp(mmd_total / len(gammas), min=0.0).item())


def save_latent_projection(latents_s1, latents_s2, output_path):
    """Project latent codes to 2D via PCA and save a scatter plot."""
    if not latents_s1 or not latents_s2:
        return None

    latents_s1 = [tensor.float() for tensor in latents_s1 if tensor.numel() > 0]
    latents_s2 = [tensor.float() for tensor in latents_s2 if tensor.numel() > 0]
    if not latents_s1 or not latents_s2:
        return None

    lat_s1 = torch.cat(latents_s1, dim=0)
    lat_s2 = torch.cat(latents_s2, dim=0)
    if lat_s1.numel() == 0 or lat_s2.numel() == 0:
        return None

    combined = torch.cat([lat_s1, lat_s2], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean
    latent_dim = centered.shape[1]
    if latent_dim == 0:
        return None

    q = min(2, latent_dim)
    try:
        _, _, v = torch.pca_lowrank(centered, q=q)
    except RuntimeError:
        # Fallback to SVD if PCA fails (e.g., low-rank issues)
        u, _, vh = torch.linalg.svd(centered, full_matrices=False)
        v = vh.t()
    components = v[:, :q]

    proj_s1 = (lat_s1 - mean) @ components
    proj_s2 = (lat_s2 - mean) @ components

    if q == 1:
        proj_s1 = torch.cat([proj_s1, torch.zeros_like(proj_s1)], dim=1)
        proj_s2 = torch.cat([proj_s2, torch.zeros_like(proj_s2)], dim=1)

    proj_s1_np = proj_s1[:, :2].detach().cpu().numpy()
    proj_s2_np = proj_s2[:, :2].detach().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(proj_s1_np[:, 0], proj_s1_np[:, 1], alpha=0.6, color='tab:blue', label='Model S1')
    ax.scatter(proj_s2_np[:, 0], proj_s2_np[:, 1], alpha=0.6, color='tab:orange', label='Model S2')
    ax.set_xlabel('PC1', fontsize=25)
    ax.set_ylabel('PC2', fontsize=25)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def add_padded_adjacency_matrix(data, n_max_nodes):
    """
    Add padded adjacency matrix to data object if not present.
    This converts sparse edge_index to dense padded adjacency matrix.
    Needed for datasets that store only edge_index to save space.
    
    Args:
        data: PyG Data object
        n_max_nodes: Target size for padding
    
    Returns:
        data with A attribute added
    """
    # Check if A exists and has correct size
    if hasattr(data, 'A') and data.A is not None:
        current_size = data.A.shape[-1]  # Last dimension size
        if current_size == n_max_nodes:
            return data  # Already has correct size
        else:
            # Need to resize - extract actual graph first, then repad
            print(f"  [Warning] Resizing adjacency from {current_size}×{current_size} to {n_max_nodes}×{n_max_nodes}")
            # Extract the actual graph (first num_nodes×num_nodes)
            num_nodes = data.num_nodes
            A = data.A[0, :num_nodes, :num_nodes]
    else:
        # Create dense adjacency matrix from edge_index
        num_nodes = data.num_nodes
        A = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
        
        if data.edge_index.numel() > 0:
            edge_index = data.edge_index
            A[edge_index[0], edge_index[1]] = 1.0
    
    # Pad to n_max_nodes
    size_diff = n_max_nodes - num_nodes
    if size_diff > 0:
        A_padded = F.pad(A, [0, size_diff, 0, size_diff])
    else:
        # If graph is larger than n_max_nodes, truncate
        A_padded = A[:n_max_nodes, :n_max_nodes]
    
    A_padded = A_padded.unsqueeze(0)  # Shape: (1, n_max_nodes, n_max_nodes)
    data.A = A_padded
    
    return data


def train_autoencoder(data_list, run_name, output_dir):
    """Train VGAE on given data."""
    print(f"\n{'='*80}")
    print(f"Training Autoencoder: {run_name}")
    print(f"{'='*80}")
    
    # Add padded adjacency matrices if not present (for space-efficient datasets)
    # Check if first data point has A attribute
    if not hasattr(data_list[0], 'A') or data_list[0].A is None:
        print(f"Adding padded adjacency matrices on-the-fly (not stored in dataset)...")
        # Get n_max_nodes from data or use N_MAX_NODES constant
        n_max = data_list[0].n_max_nodes if hasattr(data_list[0], 'n_max_nodes') else N_MAX_NODES
        data_list = [add_padded_adjacency_matrix(data, n_max) for data in data_list]
    
    # Get feature dimension
    input_feat_dim = data_list[0].x.shape[1]
    
    # Create model
    autoencoder = VariationalAutoEncoder(
        input_dim=input_feat_dim,
        hidden_dim_enc=HIDDEN_DIM_ENCODER,
        hidden_dim_dec=HIDDEN_DIM_DECODER,
        latent_dim=LATENT_DIM,
        n_layers_enc=2,
        n_layers_dec=3,
        n_max_nodes=N_MAX_NODES
    ).to(device)
    
    dataset_size = len(data_list)
    beta_value = BETA_KL_WEIGHT
    if dataset_size <= SMALL_DATASET_THRESHOLD:
        beta_value = SMALL_DATASET_KL_WEIGHT
        autoencoder.encoder.dropout = SMALL_DATASET_DROPOUT
        print(f"Small dataset detected ({dataset_size} graphs) → KL beta set to {beta_value}, encoder dropout={SMALL_DATASET_DROPOUT}")
    
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    patience_counter = 0
    best_epoch = 0
    best_val_loss_per_graph = np.inf
    checkpoint_path = output_dir / f'autoencoder_{run_name}.pth.tar'
    
    # Data loader (using 80% for train, 20% for val)
    # For small datasets (N < 10), use all data for both train and val to avoid extreme cases
    if len(data_list) < 10:
        train_data = data_list
        val_data = data_list
    else:
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train] if n_train > 0 else data_list
        val_data = data_list[n_train:] if n_train > 0 and len(data_list) > 1 else train_data
    
    # Use consistent batch size across all experiments (capped by data size)
    batch_size = min(BATCH_SIZE, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(BATCH_SIZE, len(val_data)), shuffle=False)
    
    best_val_loss = np.inf  # keep for backward compatibility
    
    edges_per_graph = N_MAX_NODES * N_MAX_NODES

    for epoch in range(1, EPOCHS_AUTOENCODER + 1):
        autoencoder.train()
        
        train_loss_all = 0
        train_count = 0
        train_recon_sum = 0
        train_kld_sum = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld = autoencoder.loss_function(data, beta=beta_value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), GRAD_CLIP)
            train_loss_all += loss.item()
            train_recon_sum += recon.item()
            train_kld_sum += kld.item()
            train_count += int(torch.max(data.batch).item()) + 1
            optimizer.step()
        
        # Validation
        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        val_recon_sum = 0
        val_kld_sum = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                loss, recon, kld = autoencoder.loss_function(data, beta=beta_value)
                val_loss_all += loss.item()
                val_recon_sum += recon.item()
                val_kld_sum += kld.item()
                val_count += int(torch.max(data.batch).item()) + 1
        
        train_loss_avg = train_loss_all / train_count if train_count else np.nan
        val_loss_avg = val_loss_all / val_count if val_count else np.nan
        train_recon_avg = train_recon_sum / train_count if train_count else np.nan
        val_recon_avg = val_recon_sum / val_count if val_count else np.nan
        train_per_edge = train_recon_avg / edges_per_graph if np.isfinite(train_recon_avg) else np.nan
        val_per_edge = val_recon_avg / edges_per_graph if np.isfinite(val_recon_avg) else np.nan

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: Train Loss: {train_loss_avg:.2f} | Val Loss: {val_loss_avg:.2f} "
                f"(per-edge train MAE {train_per_edge:.4f}, val MAE {val_per_edge:.4f})"
            )
        
        scheduler.step()
        
        # Early stopping logic
        if val_loss_avg < best_val_loss_per_graph:
            best_val_loss_per_graph = val_loss_avg
            best_val_loss = val_loss_all
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss_per_graph': best_val_loss_per_graph,
                'beta': beta_value,
            }, checkpoint_path)
            if epoch % 20 == 0:
                print('  → New best model saved!')
        else:
            patience_counter += 1
            if epoch % 20 == 0 and patience_counter > 0:
                print(f'  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')
        
        # Check if should stop early
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', best_epoch)
        best_val_loss_per_graph = checkpoint.get('best_val_loss_per_graph', best_val_loss_per_graph)
    else:
        print('Warning: Best checkpoint not found; returning last epoch weights.')

    best_val_mae = best_val_loss_per_graph / edges_per_graph if np.isfinite(best_val_loss_per_graph) else np.nan
    print(
        f"Autoencoder training complete. Best val loss/graph: {best_val_loss_per_graph:.2f}"
        f" (per-edge MAE {best_val_mae:.4f}) at epoch {best_epoch}"
    )

    # Post-training diagnostics
    def reconstruction_metrics(loader):
        if loader is None or len(loader.dataset) == 0:
            return np.nan, np.nan
        autoencoder.eval()
        total_abs = 0.0
        total_entries = 0
        total_correct = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = autoencoder(batch)
                targets = batch.A
                total_abs += torch.abs(preds - targets).sum().item()
                total_entries += targets.numel()
                total_correct += ((preds > 0.5).float() == targets).float().sum().item()
        mae = total_abs / total_entries if total_entries else np.nan
        acc = total_correct / total_entries if total_entries else np.nan
        return mae, acc

    train_mae, train_acc = reconstruction_metrics(train_loader)
    val_mae, val_acc = reconstruction_metrics(val_loader)
    print(f"  Reconstruction MAE (train): {train_mae:.4f}, accuracy: {train_acc:.4f}")
    if np.isfinite(val_mae):
        print(f"  Reconstruction MAE (val):   {val_mae:.4f}, accuracy: {val_acc:.4f}")
    
    # Return metrics along with model
    metrics = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss_per_graph,
        'best_val_mae': best_val_mae,
        'train_mae': train_mae,
        'train_acc': train_acc,
        'val_mae': val_mae,
        'val_acc': val_acc,
        'dataset_size': len(data_list),
        'beta_kl': beta_value
    }
    
    return autoencoder, metrics


def train_denoiser(autoencoder, data_list, run_name, output_dir):
    """Train latent diffusion model."""
    print(f"\n{'='*80}")
    print(f"Training Denoiser: {run_name}")
    print(f"{'='*80}")
    
    # Add padded adjacency matrices if not present (for space-efficient datasets)
    if not hasattr(data_list[0], 'A') or data_list[0].A is None:
        print(f"Adding padded adjacency matrices on-the-fly (not stored in dataset)...")
        n_max = data_list[0].n_max_nodes if hasattr(data_list[0], 'n_max_nodes') else N_MAX_NODES
        data_list = [add_padded_adjacency_matrix(data, n_max) for data in data_list]
    
    # Create denoiser
    denoise_model = DenoiseNN(
        input_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM_DENOISE,
        n_layers=3,
        n_cond=N_PROPERTIES,
        d_cond=128
    ).to(device)
    
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # Beta schedule for diffusion
    betas = linear_beta_schedule(timesteps=TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)
    
    # Data loader
    # For small datasets (N < 10), use all data for both train and val to avoid extreme cases
    if len(data_list) < 10:
        train_data = data_list
        val_data = data_list
    else:
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train] if n_train > 0 else data_list
        val_data = data_list[n_train:] if n_train > 0 and len(data_list) > 1 else train_data
    
    # Use consistent batch size across all experiments (capped by data size)
    batch_size = min(BATCH_SIZE, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(BATCH_SIZE, len(val_data)), shuffle=False)
    
    autoencoder.eval()
    best_val_loss_per_graph = np.inf
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = output_dir / f'denoise_{run_name}.pth.tar'
    
    for epoch in range(1, EPOCHS_DENOISER + 1):
        denoise_model.train()
        
        train_loss_all = 0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                x_g = autoencoder.encode(data)
            
            # Ensure stats are in correct shape [batch_size, n_properties]
            stats = data.stats[:, :N_PROPERTIES].reshape(-1, N_PROPERTIES)
            
            t = torch.randint(0, TIMESTEPS, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, stats,
                          sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                          loss_type="huber")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), GRAD_CLIP)
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()
        
        # Validation
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                
                # Ensure stats are in correct shape [batch_size, n_properties]
                stats = data.stats[:, :N_PROPERTIES].reshape(-1, N_PROPERTIES)
                
                t = torch.randint(0, TIMESTEPS, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, stats,
                              sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                              loss_type="huber")
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)
        
        train_loss_avg = train_loss_all / train_count if train_count else np.nan
        val_loss_avg = val_loss_all / val_count if val_count else np.nan
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}: Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}')
        
        scheduler.step()
        
        # Early stopping logic
        if val_loss_avg < best_val_loss_per_graph:
            best_val_loss_per_graph = val_loss_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss_per_graph': best_val_loss_per_graph
            }, checkpoint_path)
            if epoch % 10 == 0:
                print('  → New best model saved!')
        else:
            patience_counter += 1
            if epoch % 10 == 0 and patience_counter > 0:
                print(f'  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')
        
        # Check if should stop early
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        denoise_model.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', best_epoch)
        best_val_loss_per_graph = checkpoint.get('best_val_loss_per_graph', best_val_loss_per_graph)
    else:
        print('Warning: Best denoiser checkpoint not found; returning last epoch weights.')

    print(f"Denoiser training complete. Best val loss: {best_val_loss_per_graph:.4f} (epoch {best_epoch})")
    
    # Return metrics along with model and betas
    metrics = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss_per_graph,
        'dataset_size': len(data_list)
    }
    
    return denoise_model, betas, metrics


def generate_graphs(autoencoder, denoise_model, conditioning_stats, betas, num_samples=1):
    """Generate graphs and return decoded adjacencies plus latent codes used for decoding."""
    autoencoder.eval()
    denoise_model.eval()

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # Prepare conditioning (ensure correct shape)
    if isinstance(conditioning_stats, torch.Tensor):
        cond_tensor = conditioning_stats
    else:
        cond_tensor = torch.tensor(conditioning_stats, dtype=torch.float32)

    if cond_tensor.dim() == 1:
        cond_tensor = cond_tensor.unsqueeze(0)

    cond_tensor = cond_tensor.to(device)

    if cond_tensor.size(0) == 1 and num_samples > 1:
        cond_tensor = cond_tensor.repeat(num_samples, 1)
    elif cond_tensor.size(0) != num_samples:
        raise ValueError(
            f"Conditioning stats batch ({cond_tensor.size(0)}) does not match requested samples ({num_samples})."
        )

    with torch.no_grad():
        samples = sample(
            denoise_model,
            cond_tensor,
            latent_dim=LATENT_DIM,
            timesteps=TIMESTEPS,
            betas=betas,
            batch_size=num_samples
        )
        x_sample = samples[-1]
        # Use soft sampling during generation to get probabilities (not hard-thresholded yet)
        adj = autoencoder.decode_mu(x_sample, use_soft_sampling=True)

        graphs = []
        adj_matrices = []
        latent_codes = x_sample.detach().cpu()
        
        # Extract target graph sizes from conditioning stats
        # First element of stats is num_nodes
        target_sizes = cond_tensor[:, 0].cpu().numpy().astype(int)
        # Extract optional target edges/density if available
        has_edges = cond_tensor.size(1) > 1
        has_density = cond_tensor.size(1) > 2
        target_edges_arr = None
        if has_edges:
            # Guard against NaNs/negatives
            edges_raw = cond_tensor[:, 1].detach().cpu()
            edges_raw = torch.nan_to_num(edges_raw, nan=0.0)
            target_edges_arr = edges_raw.clamp(min=0).long().numpy()
        
        # We'll adaptively select edges in the n×n region to match target edge count (if available)
        for idx in range(adj.shape[0]):
            n = int(target_sizes[idx] if idx < len(target_sizes) else target_sizes[0])
            n = max(0, min(n, adj.shape[1]))
            if n <= 1:
                # Zero out and skip
                adj[idx] = 0
                continue
            # Mask out padding region outside n×n
            adj[idx, n:, :] = 0
            adj[idx, :, n:] = 0
            # Work on probabilities in the target region
            sub = adj[idx, :n, :n]
            # Use only upper triangle without diagonal
            tri = torch.triu_indices(n, n, offset=1, device=sub.device)
            vals = sub[tri[0], tri[1]]
            # Determine target edge count
            if target_edges_arr is not None:
                tgt_e = int(target_edges_arr[idx] if idx < len(target_edges_arr) else target_edges_arr[0])
            elif has_density:
                density = float(torch.nan_to_num(cond_tensor[idx, 2], nan=0.0).item())
                max_edges = n * (n - 1) // 2
                tgt_e = int(max(0, min(max_edges, round(density * max_edges))))
            else:
                # Fallback: small positive ratio of possible edges
                max_edges = n * (n - 1) // 2
                tgt_e = max(1, int(0.02 * max_edges))
            max_edges = n * (n - 1) // 2
            tgt_e = max(0, min(tgt_e, max_edges))
            # If probabilities are all zero, create a minimal connected structure as fallback
            if torch.all(vals <= 0):
                # Build a simple chain to avoid empty graph
                sub_bin = torch.zeros_like(sub)
                for u in range(n - 1):
                    sub_bin[u, u + 1] = 1.0
                    sub_bin[u + 1, u] = 1.0
                # If chain has more edges than target, it's fine for now; we'll keep minimal structure
                adj[idx, :n, :n] = sub_bin
                continue
            # Select top-k edges according to probabilities
            k = tgt_e
            if k <= 0:
                adj[idx, :n, :n] = 0
                continue
            k = int(min(k, vals.numel()))
            topk = torch.topk(vals, k=k, largest=True, sorted=False)
            chosen = torch.zeros_like(vals)
            chosen[topk.indices] = 1.0
            # Scatter back into a binary adjacency
            sub_bin = torch.zeros_like(sub)
            sub_bin[tri[0], tri[1]] = chosen
            sub_bin = sub_bin + sub_bin.transpose(0, 1)
            # Assign back
            adj[idx, :n, :n] = sub_bin
        
        for idx in range(num_samples):
            adj_np = adj[idx].detach().cpu().numpy()
            
            # Extract only the relevant n×n submatrix based on target size
            # The decoder always outputs N_MAX_NODES×N_MAX_NODES, but actual graph is smaller
            n_target = target_sizes[idx] if idx < len(target_sizes) else target_sizes[0]
            n_target = min(n_target, adj_np.shape[0])  # Safety check
            
            # Extract submatrix
            adj_np_trimmed = adj_np[:n_target, :n_target]
            
            # Debug: Print detailed info for first generation
            if idx == 0 and len(graphs) == 0:
                edges_full = int((adj_np > 0.5).sum())
                edges_trimmed = int((adj_np_trimmed > 0.5).sum())
                
                # Check where edges are located in the full matrix
                edges_in_target_region = int((adj_np[:n_target, :n_target] > 0.5).sum())
                edges_outside_target = edges_full - edges_in_target_region
                
                # print(f"  [Generation Debug] Decoder output shape: {adj_np.shape}, Target size: {n_target}")
                # print(f"    Full matrix: {edges_full} edges")
                # print(f"    Target region [0:{n_target}, 0:{n_target}]: {edges_in_target_region} edges")
                # print(f"    Outside target region: {edges_outside_target} edges")
                # print(f"    Value range: [{adj_np.min():.4f}, {adj_np.max():.4f}]")
                # print(f"    Unique values: {np.unique(adj_np)[:10]}")  # Show first 10 unique values
                
                if edges_trimmed == 0:
                    print(f"    [WARNING] Generated graph is EMPTY!")
                    # Check if it's truly all zeros or just below threshold
                    max_val_in_target = adj_np[:n_target, :n_target].max()
                    print(f"    Max value in target region: {max_val_in_target:.6f}")
                    if max_val_in_target == 0:
                        print(f"    → Decoder output is completely zero in target region!")
                    else:
                        print(f"    → Decoder has non-zero values but all below 0.5 threshold")
            
            graphs.append(construct_nx_from_adj(adj_np_trimmed))
            adj_matrices.append(adj_np_trimmed)

    return graphs, adj_matrices, latent_codes


def compute_wl_similarity(G1, G2):
    """
    Compute Weisfeiler-Lehman kernel similarity between two graphs.
    
    Uses node degree as labels (topology-derived labels) since generated graphs
    don't have semantic node labels. When node feature vectors are available
    (stored under the `feature_vector` attribute), we augment the topology label
    with a coarse discretization of the first few feature dimensions so the WL
    kernel is sensitive to feature homophily as well.
    """
    # Handle empty graphs (can happen with generation failures)
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        # Both empty: This should NOT be treated as perfect similarity!
        # Empty graphs indicate generation failure, not actual similarity
        if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
            return 0.0  # Changed from 1.0 - empty graphs are invalid, not identical
        # One empty, one not: completely different
        else:
            return 0.0
    
    try:
        # Work on copies to avoid mutating upstream graphs
        G1_local = G1.copy()
        G2_local = G2.copy()

        def feature_signature(node_data):
            vec = node_data.get('feature_vector')
            if vec is None:
                return 'feat:none'
            vec = np.asarray(vec, dtype=float)
            if vec.size == 0:
                return 'feat:none'
            capped = vec[:4]
            signature = []
            for value in capped:
                if value < -0.25:
                    signature.append('n')
                elif value > 0.25:
                    signature.append('p')
                else:
                    signature.append('z')
            if vec.size > 4:
                mean_rest = float(np.mean(vec[4:]))
                if mean_rest < -0.25:
                    signature.append('rN')
                elif mean_rest > 0.25:
                    signature.append('rP')
                else:
                    signature.append('rZ')
            return 'feat:' + ''.join(signature)

        # Check if both graphs have features - if not, use only topology
        g1_has_features = any('feature_vector' in G1_local.nodes[n] for n in G1_local.nodes())
        g2_has_features = any('feature_vector' in G2_local.nodes[n] for n in G2_local.nodes())
        use_features = g1_has_features and g2_has_features
        
        def annotate_graph(G_local, use_feat):
            for node in G_local.nodes():
                base_label = int(G_local.degree(node))
                if use_feat:
                    feat_sig = feature_signature(G_local.nodes[node])
                    G_local.nodes[node]['label'] = f"{base_label}|{feat_sig}"
                else:
                    # Use only topology (degree) when features unavailable
                    G_local.nodes[node]['label'] = str(base_label)
            return G_local

        G1_local = annotate_graph(G1_local, use_features)
        G2_local = annotate_graph(G2_local, use_features)
        
        # Convert to grakel format
        graphs_pair = graph_from_networkx([G1_local, G2_local], node_labels_tag='label')
        
        # Compute WL kernel
        wl_kernel = WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)
        K = wl_kernel.fit_transform(graphs_pair)
        
        # Similarity is off-diagonal element
        similarity = K[0, 1]
        return similarity
    except Exception as e:
        # Fallback: simple edge overlap
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        if len(edges1) == 0 and len(edges2) == 0:
            return 1.0
        union = len(edges1.union(edges2))
        intersection = len(edges1.intersection(edges2))
        return intersection / union if union > 0 else 0.0


def compute_statistics_distance(G1, G2):
    """Compute MSE and MAE between graph statistics."""
    try:
        stats1 = gen_stats(G1)
        stats2 = gen_stats(G2)
        
        # Convert to numpy arrays (first 15 features)
        stats1 = np.array(stats1[:N_PROPERTIES])
        stats2 = np.array(stats2[:N_PROPERTIES])
        
        mse = np.mean((stats1 - stats2) ** 2)
        mae = np.mean(np.abs(stats1 - stats2))
        
        return mse, mae
    except Exception as e:
        print(f"Warning: Statistics computation failed ({e})")
        return np.nan, np.nan


def find_closest_graph_in_training(generated_G, training_graphs_nx):
    """Find the most similar graph in training set to the generated graph."""
    best_sim = -1
    closest_G = None
    
    for G_train in training_graphs_nx:
        # Compute similarity
        sim = compute_wl_similarity(generated_G, G_train)
        
        if sim > best_sim:
            best_sim = sim
            closest_G = G_train
    
    return closest_G, best_sim


def compute_dataset_complexity(data_list):
    """
    Compute complexity metrics for the dataset to track memorization vs generalization threshold.
    
    Returns:
        dict: Complexity metrics including avg nodes, edges, variance, etc.
    """
    num_nodes = []
    num_edges = []
    avg_degrees = []
    
    for data in data_list:
        # Use num_nodes and edge_index instead of A (which may not exist in space-efficient datasets)
        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2 if data.edge_index.numel() > 0 else 0  # Undirected graph
        
        num_nodes.append(n_nodes)
        num_edges.append(n_edges)
        if n_nodes > 0:
            avg_degrees.append(2 * n_edges / n_nodes)
    
    complexity_metrics = {
        'avg_nodes': np.mean(num_nodes) if num_nodes else 0,
        'std_nodes': np.std(num_nodes) if num_nodes else 0,
        'avg_edges': np.mean(num_edges) if num_edges else 0,
        'std_edges': np.std(num_edges) if num_edges else 0,
        'avg_degree': np.mean(avg_degrees) if avg_degrees else 0,
        'std_degree': np.std(avg_degrees) if avg_degrees else 0,
        'total_possible_edges': np.mean([n * (n - 1) / 2 for n in num_nodes]) if num_nodes else 0,
    }
    
    return complexity_metrics


def run_experiment_for_N(N, split_config, output_dir):
    """
    Run training experiment for a specific training size N.
    Focus on training models and logging metrics to understand memorization vs generalization.
    
    Returns:
        Dictionary with training metrics, complexity ratios, and model checkpoints
    """
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: N = {N}")
    print(f"{'#'*80}")
    
    # Create splits
    S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices = create_splits(split_config, N)
    stats_cache_path = split_config.get('stats_cache_path')
    if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) == len(test_graphs) and stats_cache_path:
        print(f"  Using conditioning cache: {stats_cache_path}")
    
    # Create output directories for this N
    exp_dir = output_dir / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check distribution parity between S1 and S2 statistics
    distribution_summary = perform_distribution_checks(S1, S2, exp_dir, N)
    
    # Compute complexity metrics for the training data
    print(f"\n--- Computing Complexity Metrics ---")
    s1_complexity = compute_dataset_complexity(S1)
    s2_complexity = compute_dataset_complexity(S2)
    
    # Compute complexity/dataset_size ratio
    # Using average number of possible edges as a proxy for task complexity
    s1_complexity_ratio = s1_complexity['total_possible_edges'] / N if N > 0 else np.inf
    s2_complexity_ratio = s2_complexity['total_possible_edges'] / N if N > 0 else np.inf
    
    print(f"S1 Complexity Metrics:")
    print(f"  Avg nodes: {s1_complexity['avg_nodes']:.2f} ± {s1_complexity['std_nodes']:.2f}")
    print(f"  Avg edges: {s1_complexity['avg_edges']:.2f} ± {s1_complexity['std_edges']:.2f}")
    print(f"  Avg degree: {s1_complexity['avg_degree']:.2f} ± {s1_complexity['std_degree']:.2f}")
    print(f"  Complexity/Dataset ratio: {s1_complexity_ratio:.4f}")
    
    print(f"\nS2 Complexity Metrics:")
    print(f"  Avg nodes: {s2_complexity['avg_nodes']:.2f} ± {s2_complexity['std_nodes']:.2f}")
    print(f"  Avg edges: {s2_complexity['avg_edges']:.2f} ± {s2_complexity['std_edges']:.2f}")
    print(f"  Avg degree: {s2_complexity['avg_degree']:.2f} ± {s2_complexity['std_degree']:.2f}")
    print(f"  Complexity/Dataset ratio: {s2_complexity_ratio:.4f}")
    
    s1_min_idx = int(np.min(S1_indices))
    s1_max_idx = int(np.max(S1_indices))
    s2_min_idx = int(np.min(S2_indices))
    s2_max_idx = int(np.max(S2_indices))

    # Train Model S1
    print(f"\n--- Training Model S1 (original indices {s1_min_idx}–{s1_max_idx}) ---")
    autoencoder_S1, ae_s1_metrics = train_autoencoder(S1, f"S1_N{N}", exp_dir)
    denoise_S1, betas, dn_s1_metrics = train_denoiser(autoencoder_S1, S1, f"S1_N{N}", exp_dir)
    
    # Train Model S2
    print(f"\n--- Training Model S2 (original indices {s2_min_idx}–{s2_max_idx}) ---")
    autoencoder_S2, ae_s2_metrics = train_autoencoder(S2, f"S2_N{N}", exp_dir)
    denoise_S2, _, dn_s2_metrics = train_denoiser(autoencoder_S2, S2, f"S2_N{N}", exp_dir)
    
    print(f"\nTraining Complete for N={N}")
    print(f"  S1 Autoencoder - Best Val Loss: {ae_s1_metrics['best_val_loss']:.4f} at epoch {ae_s1_metrics['best_epoch']}")
    print(f"  S1 Denoiser - Best Val Loss: {dn_s1_metrics['best_val_loss']:.4f} at epoch {dn_s1_metrics['best_epoch']}")
    print(f"  S2 Autoencoder - Best Val Loss: {ae_s2_metrics['best_val_loss']:.4f} at epoch {ae_s2_metrics['best_epoch']}")
    print(f"  S2 Denoiser - Best Val Loss: {dn_s2_metrics['best_val_loss']:.4f} at epoch {dn_s2_metrics['best_epoch']}")
    
    # Generate graphs and compute WL similarities
    print(f"\n--- Generating Graphs and Computing WL Similarities ---")
    generalization_scores = []
    memorization_scores = []
    
    # Precompute NetworkX versions of S1 training graphs for memorization checks
    S1_training_graphs_nx = []
    for data in S1:
        # Create adjacency matrix from edge_index if A doesn't exist
        if hasattr(data, 'A') and data.A is not None:
            adj = data.A[0].cpu().numpy()
        else:
            n_nodes = data.num_nodes
            adj = np.zeros((n_nodes, n_nodes))
            if data.edge_index.numel() > 0:
                edge_index = data.edge_index.cpu().numpy()
                adj[edge_index[0], edge_index[1]] = 1.0
        
        features = data.x.detach().cpu().numpy() if hasattr(data, 'x') else None
        S1_training_graphs_nx.append(construct_nx_from_adj(adj, node_features=features))
    
    for i, test_graph in enumerate(tqdm(test_graphs, desc=f"Generating & evaluating (N={N})")):
        # Extract conditioning statistics
        if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) > i:
            c_test = test_stats_cache[i:i+1]
        else:
            c_test = test_graph.stats[:, :N_PROPERTIES]
        
        # Generate from both models with the SAME random seed for fair comparison
        # This ensures both models start from identical noise
        torch.manual_seed(42 + i)  # Different seed per test graph, but same for both models
        G1_samples, _, _ = generate_graphs(autoencoder_S1, denoise_S1, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)
        
        torch.manual_seed(42 + i)  # Reset to same seed for S2
        G2_samples, _, _ = generate_graphs(autoencoder_S2, denoise_S2, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)
        
        # Compute generalization: WL similarity between paired samples from S1 and S2
        for sample_idx in range(NUM_SAMPLES_PER_CONDITION):
            sim_generalization = compute_wl_similarity(G1_samples[sample_idx], G2_samples[sample_idx])
            generalization_scores.append(sim_generalization)
        
        # Compute memorization: WL similarity of first S1 sample to closest training graph
        _, sim_memorization = find_closest_graph_in_training(G1_samples[0], S1_training_graphs_nx)
        memorization_scores.append(sim_memorization)
    
    print(f"\nWL Similarity Results for N={N}:")
    print(f"  Generalization (S1 vs S2): {np.mean(generalization_scores):.4f} ± {np.std(generalization_scores):.4f}")
    print(f"  Memorization (S1 vs closest training): {np.mean(memorization_scores):.4f} ± {np.std(memorization_scores):.4f}")
    
    return {
        'N': N,
        'exp_dir': exp_dir,
        'distribution_summary': distribution_summary,
        'S1_indices': S1_indices,
        'S2_indices': S2_indices,
        'test_indices': test_indices,
        's1_complexity': s1_complexity,
        's2_complexity': s2_complexity,
        's1_complexity_ratio': s1_complexity_ratio,
        's2_complexity_ratio': s2_complexity_ratio,
        'ae_s1_metrics': ae_s1_metrics,
        'ae_s2_metrics': ae_s2_metrics,
        'dn_s1_metrics': dn_s1_metrics,
        'dn_s2_metrics': dn_s2_metrics,
        'generalization_scores': generalization_scores,
        'memorization_scores': memorization_scores,
    }


def visualize_single_experiment(results):
    """Save quick-look visualizations for a single N immediately after the run."""
    exp_dir = results['exp_dir']
    quick_dir = exp_dir / "quicklook"
    quick_dir.mkdir(exist_ok=True)

    N = results['N']
    
    # 1. WL Similarity Histogram (separate figure as requested)
    gen_scores = results.get('generalization_scores', [])
    mem_scores = results.get('memorization_scores', [])
    
    if len(gen_scores) > 0 and len(mem_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mem_scores, bins=20, alpha=0.7, color='orange', range=(0, 1), 
                label='Memorization (vs training set)', edgecolor='black', linewidth=0.5)
        ax.hist(gen_scores, bins=20, alpha=0.7, color='blue', range=(0, 1), 
                label='Generalization (S1 vs S2)', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('WL Kernel Similarity', fontsize=25)
        ax.set_ylabel('Frequency', fontsize=25)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=18, loc='upper left')
        ax.set_xlim(0, 1)
        
        # Add mean lines
        if len(gen_scores) > 0:
            ax.axvline(np.mean(gen_scores), color='blue', linestyle='--', linewidth=2, alpha=0.8)
        if len(mem_scores) > 0:
            ax.axvline(np.mean(mem_scores), color='orange', linestyle='--', linewidth=2, alpha=0.8)
        
        plt.tight_layout()
        hist_path = quick_dir / f"wl_similarity_histogram_N{N}.png"
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"WL similarity histogram saved for N={N}: {hist_path}")
    
    # 2. Training metrics plot (separate figure)
    ae_s1 = results['ae_s1_metrics']
    ae_s2 = results['ae_s2_metrics']
    dn_s1 = results['dn_s1_metrics']
    dn_s2 = results['dn_s2_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Autoencoder validation loss
    ax = axes[0, 0]
    ax.bar(['S1', 'S2'], [ae_s1['best_val_loss'], ae_s2['best_val_loss']], color=['#3498db', '#e67e22'])
    ax.set_ylabel('Best Val Loss', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    
    # Autoencoder validation MAE
    ax = axes[0, 1]
    ax.bar(['S1', 'S2'], [ae_s1['best_val_mae'], ae_s2['best_val_mae']], color=['#3498db', '#e67e22'])
    ax.set_ylabel('Best Val MAE', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    
    # Denoiser validation loss
    ax = axes[1, 0]
    ax.bar(['S1', 'S2'], [dn_s1['best_val_loss'], dn_s2['best_val_loss']], color=['#3498db', '#e67e22'])
    ax.set_ylabel('Denoiser Best Val Loss', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    
    # Complexity ratio
    ax = axes[1, 1]
    ax.bar(['S1', 'S2'], [results['s1_complexity_ratio'], results['s2_complexity_ratio']], 
           color=['#3498db', '#e67e22'])
    ax.set_ylabel('Complexity/Dataset Ratio', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    metrics_path = quick_dir / f"training_metrics_N{N}.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training metrics saved for N={N}: {metrics_path}")


def visualize_results(all_results, output_dir):
    """
    Create visualizations for training metrics across different N values.
    Focus on complexity/dataset ratios and convergence metrics.
    """
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")
    print("\nIMPORTANT: All experiments used identical hyperparameters.")
    print("Only the training set size (N) was varied.")
    print(f"This ensures fair comparison with no hidden confounders.\n")
    
    # Create figure directory
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    N_vals = []
    s1_ae_losses = []
    s2_ae_losses = []
    s1_dn_losses = []
    s2_dn_losses = []
    s1_complexity_ratios = []
    s2_complexity_ratios = []
    
    for N in N_VALUES:
        if N in all_results:
            results = all_results[N]
            N_vals.append(N)
            s1_ae_losses.append(results['ae_s1_metrics']['best_val_loss'])
            s2_ae_losses.append(results['ae_s2_metrics']['best_val_loss'])
            s1_dn_losses.append(results['dn_s1_metrics']['best_val_loss'])
            s2_dn_losses.append(results['dn_s2_metrics']['best_val_loss'])
            s1_complexity_ratios.append(results['s1_complexity_ratio'])
            s2_complexity_ratios.append(results['s2_complexity_ratio'])
    
    # Figure 1: Convergence curves - Autoencoder
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(N_vals, s1_ae_losses, 'o-', color='#3498db', linewidth=2, markersize=8, label='S1')
    ax.plot(N_vals, s2_ae_losses, 's-', color='#e67e22', linewidth=2, markersize=8, label='S2')
    ax.set_xlabel('Dataset Size (N)', fontsize=25)
    ax.set_ylabel('Autoencoder Val Loss', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'autoencoder_convergence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'autoencoder_convergence.png'}")
    plt.close()
    
    # Figure 2: Convergence curves - Denoiser
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(N_vals, s1_dn_losses, 'o-', color='#3498db', linewidth=2, markersize=8, label='S1')
    ax.plot(N_vals, s2_dn_losses, 's-', color='#e67e22', linewidth=2, markersize=8, label='S2')
    ax.set_xlabel('Dataset Size (N)', fontsize=25)
    ax.set_ylabel('Denoiser Val Loss', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'denoiser_convergence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'denoiser_convergence.png'}")
    plt.close()
    
    # Figure 3: Complexity/Dataset Ratio (separate figure)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(N_vals, s1_complexity_ratios, 'o-', color='#3498db', linewidth=2, markersize=8, label='S1')
    ax.plot(N_vals, s2_complexity_ratios, 's-', color='#e67e22', linewidth=2, markersize=8, label='S2')
    ax.set_xlabel('Dataset Size (N)', fontsize=25)
    ax.set_ylabel('Complexity/Dataset Ratio', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'complexity_ratio.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'complexity_ratio.png'}")
    plt.close()
    
    # Figure 4: WL Similarity comparison across N (separate figure)
    gen_means = []
    gen_stds = []
    mem_means = []
    mem_stds = []
    
    for N in N_VALUES:
        if N in all_results:
            results = all_results[N]
            gen_scores = results.get('generalization_scores', [])
            mem_scores = results.get('memorization_scores', [])
            
            if len(gen_scores) > 0:
                gen_means.append(np.mean(gen_scores))
                gen_stds.append(np.std(gen_scores))
            else:
                gen_means.append(np.nan)
                gen_stds.append(np.nan)
            
            if len(mem_scores) > 0:
                mem_means.append(np.mean(mem_scores))
                mem_stds.append(np.std(mem_scores))
            else:
                mem_means.append(np.nan)
                mem_stds.append(np.nan)
    
    if any(~np.isnan(gen_means)) and any(~np.isnan(mem_means)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(N_vals, gen_means, yerr=gen_stds, marker='o', color='blue', 
                    linewidth=2, markersize=8, capsize=5, label='Generalization (S1 vs S2)')
        ax.errorbar(N_vals, mem_means, yerr=mem_stds, marker='s', color='orange', 
                    linewidth=2, markersize=8, capsize=5, label='Memorization (vs training)')
        ax.set_xlabel('Dataset Size (N)', fontsize=25)
        ax.set_ylabel('WL Kernel Similarity', fontsize=25)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=18)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'wl_similarity_convergence.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_dir / 'wl_similarity_convergence.png'}")
        plt.close()
    
    # Figure 3: Mean similarity vs N
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_vals = []
    gen_means = []
    gen_stds = []
    mem_means = []
    mem_stds = []



def save_summary(all_results, output_dir):
    """Save numerical summary of training results and complexity ratios."""
    summary_path = output_dir / "experiment_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Graph Generation: Memorization to Generalization Training Study\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: labelhomophily0.5_10nodes_graphs.pkl (2100 total: 1000 S1 + 1000 S2 + 100 test)\n")
        f.write(f"Training sizes tested: {N_VALUES}\n\n")
        
        f.write("Hyperparameters (constant across all N):\n")
        f.write("-"*80 + "\n")
        f.write(f"Autoencoder epochs: {EPOCHS_AUTOENCODER}\n")
        f.write(f"Denoiser epochs: {EPOCHS_DENOISER}\n")
        f.write(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Latent dimension: {LATENT_DIM}\n")
        f.write(f"Beta KL weight: {BETA_KL_WEIGHT}\n\n")
        
        f.write("Training Results Summary:\n")
        f.write("-"*180 + "\n")
        f.write(f"{'N':<8} {'AE_S1_Loss':<12} {'AE_S2_Loss':<12} {'DN_S1_Loss':<12} {'DN_S2_Loss':<12} "
                f"{'Comp_Ratio':<12} {'Gen_WL_Mean':<12} {'Gen_WL_Std':<12} {'Mem_WL_Mean':<12} {'Mem_WL_Std':<12}\n")
        f.write("-"*180 + "\n")
        
        for N in N_VALUES:
            if N in all_results:
                results = all_results[N]
                ae_s1 = results['ae_s1_metrics']
                ae_s2 = results['ae_s2_metrics']
                dn_s1 = results['dn_s1_metrics']
                dn_s2 = results['dn_s2_metrics']
                
                gen_scores = results.get('generalization_scores', [])
                mem_scores = results.get('memorization_scores', [])
                gen_mean = np.mean(gen_scores) if len(gen_scores) > 0 else np.nan
                gen_std = np.std(gen_scores) if len(gen_scores) > 0 else np.nan
                mem_mean = np.mean(mem_scores) if len(mem_scores) > 0 else np.nan
                mem_std = np.std(mem_scores) if len(mem_scores) > 0 else np.nan
                
                avg_complexity_ratio = (results['s1_complexity_ratio'] + results['s2_complexity_ratio']) / 2
                
                f.write(f"{N:<8} {ae_s1['best_val_loss']:<12.4f} {ae_s2['best_val_loss']:<12.4f} "
                       f"{dn_s1['best_val_loss']:<12.4f} {dn_s2['best_val_loss']:<12.4f} "
                       f"{avg_complexity_ratio:<12.4f} {gen_mean:<12.4f} {gen_std:<12.4f} "
                       f"{mem_mean:<12.4f} {mem_std:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nComplexity Metrics:\n")
        f.write("-"*80 + "\n")
        f.write("Complexity/Dataset Ratio = (Avg possible edges per graph) / N\n")
        f.write("Higher ratio → Model needs to learn more per sample (memorization)\n")
        f.write("Lower ratio → More redundancy in training data (generalization)\n\n")
        
        f.write("\nWL Similarity Interpretation:\n")
        f.write("-"*80 + "\n")
        f.write("Gen_WL_Mean: Average WL similarity between graphs generated by S1 and S2\n")
        f.write("Mem_WL_Mean: Average WL similarity between generated graphs and closest training graph\n")
        f.write("\nExpected Behavior:\n")
        f.write("- Memorization regime (small N):\n")
        f.write("  * High Mem_WL (generated graphs similar to training set)\n")
        f.write("  * Low Gen_WL (S1 and S2 generate different graphs)\n")
        f.write("- Generalization regime (large N):\n")
        f.write("  * Gen_WL converges toward Mem_WL\n")
        f.write("  * Both models learn similar distributions\n")
        f.write("\nHypothesis:\n")
        f.write("-"*80 + "\n")
        f.write("- Small N (high complexity/dataset ratio): Models should memorize\n")
        f.write("- Large N (low complexity/dataset ratio): Models should generalize\n")
        f.write("- Transition occurs when Gen_WL ≈ Mem_WL (models converge)\n")
        f.write("- Critical ratio marks memorization→generalization transition\n")
    
    print(f"\nSaved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Graph Generation Convergence Study')
    parser.add_argument('--data-path', type=str, 
                       default='data/labelhomophily0.5_10nodes_graphs.pkl',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, 
                       default='outputs/nodesize_study',
                       help='Output directory')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom run name')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{args.run_name}_{timestamp}"
    else:
        run_id = f"convergence_{timestamp}"
    
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Graph Generation: Memorization to Generalization Transition Study")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Training sizes: {N_VALUES}")
    print(f"{'='*80}\n")
    
    # Load dataset
    data_list = load_dataset(args.data_path)
    
    if len(data_list) < TEST_SET_SIZE:
        print(f"Warning: Expected at least {TEST_SET_SIZE} graphs for test set, found {len(data_list)}")
    
    cache_dir = Path(args.data_path).parent / "cache"
    cache_path = cache_dir / f"{Path(args.data_path).stem}_test_stats_seed{SPLIT_SEED}_size{TEST_SET_SIZE}.pt"

    split_config = shuffle_and_split_dataset(
        data_list,
        test_size=TEST_SET_SIZE,
        seed=SPLIT_SEED,
        stats_cache_path=cache_path
    )

    print("\nShuffled dataset split summary:")
    print(f"  Total graphs: {len(data_list)}")
    print(f"  S1 pool size: {len(split_config['S1_pool'])}")
    print(f"  S2 pool size: {len(split_config['S2_pool'])}")
    print(f"  Test pool size: {len(split_config['test_graphs'])} (held-out conditioning set)")
    
    # Run experiments for each N
    all_results = {}
    
    for N in N_VALUES:
        print(f"\n{'*'*80}")
        print(f"Starting experiment {len(all_results)+1}/{len(N_VALUES)} (N={N})")
        print(f"{'*'*80}")
        
        results = run_experiment_for_N(N, split_config, output_dir)
        all_results[N] = results

        visualize_single_experiment(results)
        
        # Save intermediate results
        import pickle
        with open(output_dir / f'results_N{N}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nCompleted {len(all_results)}/{len(N_VALUES)} experiments")
    
    # Create aggregate visualizations
    visualize_results(all_results, output_dir)
    
    # Save summary
    save_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Key outputs:")
    print(f"  - Autoencoder convergence: {output_dir}/figures/autoencoder_convergence.png")
    print(f"  - Denoiser convergence: {output_dir}/figures/denoiser_convergence.png")
    print(f"  - Complexity ratio plot: {output_dir}/figures/complexity_ratio.png")
    print(f"  - Summary: {output_dir}/experiment_summary.txt")
    print(f"\nTrained models saved in: {output_dir}/N_*/")
    print(f"\nNext steps:")
    print(f"  1. Generate graphs using the trained models")
    print(f"  2. Analyze memorization vs generalization based on complexity ratios")
    print(f"  3. Compare generated graphs between S1 and S2 models")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
