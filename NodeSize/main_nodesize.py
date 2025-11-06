"""
Graph Generation: Memorization to Generalization Study Across Graph Complexities

This script studies how graph complexity (number of nodes n) affects the transition
from memorization to generalization with FIXED training set size N=1000.

Terminology:
- S1, S2: Non-overlapping training subsets (1000 graphs each from total 2100)
- Gen1, Gen2: Generator models (VGAE+LDM) trained on S1 and S2 respectively
- n: Number of nodes per graph (complexity: 10, 20, 50, 100, 500, 1000)
- N: Training dataset size = 1000 (FIXED - always use full S1/S2)

Experiment:
- For each graph complexity n (10, 20, 50, 100, 500, 1000 nodes):
  - Load dataset: labelhomophily0.5_{n}nodes_graphs.pkl (2100 graphs)
  - Split: S1 (1000 graphs), S2 (1000 graphs), Test (100 graphs)
  - Train Gen1 on all 1000 graphs from S1
  - Train Gen2 on all 1000 graphs from S2
  - Generate graphs with both models using same random seed
  - Compare: Gen1 vs Gen2 (generalization) vs training set (memorization)
- Create histogram showing memorization→generalization as n increases

Hypothesis:
- Small n (simple graphs): Models generalize (Gen1 ≈ Gen2)
- Large n (complex graphs): Models memorize (Gen1 ≠ Gen2, each similar to own training set)
- With fixed N=1000, complexity/dataset ratio increases with n
- Transition point reveals when task complexity overwhelms available data
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import pandas as pd
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import (linear_beta_schedule, construct_nx_from_adj, 
                   eval_autoencoder, gen_stats)
from grakel import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

# Import functions from main_comparison.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from main_comparison import (
    compute_wl_similarity, find_closest_graph_in_training,
    shuffle_and_split_dataset, create_splits,
    perform_distribution_checks, stack_stats, ks_2samp
)

# Configuration
FIXED_N = 2500  # Fixed training set size - always use 1000 graphs from S1 and S2
NODE_SIZES = [5,10,20,30,50,100,500]  # Graph complexities to test (loop over n)
TEST_SET_SIZE = 100  # Conditioning graphs
SPLIT_SEED = 42

# Training hyperparameters (constant across all experiments)
EPOCHS_AUTOENCODER = 100
EPOCHS_DENOISER = 100
EARLY_STOPPING_PATIENCE = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
GRAD_CLIP = 1.0
LATENT_DIM = 32
HIDDEN_DIM_ENCODER = 32
HIDDEN_DIM_DECODER = 64
HIDDEN_DIM_DENOISE = 512
N_MAX_NODES = 500  # Maximum nodes (capped for computational feasibility)
N_PROPERTIES = 15  # Updated from 18 to match labelhomgenerator (no structural/feature homophily)
TIMESTEPS = 100
NUM_SAMPLES_PER_CONDITION = 5
K_NEAREST = 50  # shortlist size for k-nearest training comparisons

BETA_KL_WEIGHT = 0.05
SMALL_DATASET_THRESHOLD = 50
SMALL_DATASET_KL_WEIGHT = 0.01
SMALL_DATASET_DROPOUT = 0.1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --- Untrained-GNN embedding helpers (GIN/GCN) ---
def _nx_to_pyg_data(G: nx.Graph):
    """Convert nx.Graph to PyG Data with simple degree-based features [deg/max_deg, 1.0]."""
    from torch_geometric.data import Data
    n = G.number_of_nodes()
    if n == 0:
        x = torch.zeros((0, 2), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch = torch.zeros((0,), dtype=torch.long)
        return Data(x=x, edge_index=edge_index), batch
    nodes = list(G.nodes())
    idx_map = {u: i for i, u in enumerate(nodes)}
    max_deg = max(1, max((G.degree(u) for u in nodes), default=1))
    feats = []
    for u in nodes:
        d = G.degree(u)
        feats.append([d / max_deg, 1.0])
    x = torch.tensor(feats, dtype=torch.float32)
    edges = []
    for u, v in G.edges():
        iu, iv = idx_map[u], idx_map[v]
        edges.append((iu, iv))
        edges.append((iv, iu))
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros((n,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index), batch


class _RandomGIN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=64, layers=3, seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(last, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.layers.append(GINConv(mlp))
            last = hidden
        self.head = nn.Linear(last, out_dim)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
        g = global_mean_pool(x, batch)
        z = self.head(g)
        return z


def _embed_graphs(model: nn.Module, graphs):
    embs = []
    model.eval()
    with torch.no_grad():
        for G in graphs:
            data, batch = _nx_to_pyg_data(G)
            if data.x.size(0) == 0:
                z = torch.zeros((1, model.head.out_features), dtype=torch.float32)
            else:
                z = model(data, batch)
            embs.append(z.squeeze(0).cpu().numpy())
    if len(embs) == 0:
        return np.zeros((0, getattr(model.head, 'out_features', 64)), dtype=np.float32)
    return np.vstack(embs)


def _cosine_matrix(A: np.ndarray, B: np.ndarray):
    """Row-wise cosine similarities between all rows in A and all rows in B.
    Returns matrix [len(A), len(B)]."""
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_n @ B_n.T


# --- WL helpers: force degree-only similarity regardless of available features ---
def _strip_features(G):
    """Return a copy of G with any 'feature_vector' removed from nodes."""
    Gc = G.copy()
    for n in Gc.nodes:
        if 'feature_vector' in Gc.nodes[n]:
            try:
                del Gc.nodes[n]['feature_vector']
            except Exception:
                Gc.nodes[n]['feature_vector'] = None
    return Gc


def wl_similarity_degree_only(G1, G2):
    """Compute WL similarity using degree-only labels by stripping node features."""
    return compute_wl_similarity(_strip_features(G1), _strip_features(G2))


def find_closest_graph_in_training_degree_only(generated_G, training_graphs_nx):
    """Nearest neighbor in training under degree-only WL similarity."""
    best_sim = -1.0
    closest_G = None
    g = _strip_features(generated_G)
    for G_train in training_graphs_nx:
        sim = compute_wl_similarity(g, _strip_features(G_train))
        if sim > best_sim:
            best_sim = sim
            closest_G = G_train
    return closest_G, best_sim


def _degree_histogram_normalized(G, n_bins):
    """Return normalized degree histogram of length n_bins (0..n_bins-1)."""
    hist = np.zeros(n_bins, dtype=np.float32)
    if G.number_of_nodes() == 0:
        return hist
    for _, deg in G.degree():
        b = int(deg)
        if b >= n_bins:
            b = n_bins - 1
        hist[b] += 1.0
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return hist


def _precompute_degree_hists(training_graphs_nx, n_bins):
    return np.stack([_degree_histogram_normalized(G, n_bins) for G in training_graphs_nx], axis=0) if len(training_graphs_nx) > 0 else np.zeros((0, n_bins), dtype=np.float32)


def _k_nearest_indices_by_deg_hist(generated_G, train_hists, n_bins, k):
    """Return indices of k nearest training graphs by L1 distance on degree histograms."""
    if train_hists.shape[0] == 0:
        return []
    g_hist = _degree_histogram_normalized(generated_G, n_bins)
    dists = np.sum(np.abs(train_hists - g_hist[None, :]), axis=1)
    k_eff = int(min(max(k, 1), train_hists.shape[0]))
    idxs = np.argpartition(dists, kth=k_eff-1)[:k_eff]
    # sort these k by distance for determinism
    idxs = idxs[np.argsort(dists[idxs])]
    return idxs.tolist()


def find_closest_graph_in_training_degree_only_k(generated_G, training_graphs_nx, train_hists, n_bins, k):
    """Find best match using degree-only WL within k-nearest by degree-hist prefilter."""
    if len(training_graphs_nx) == 0:
        return None, 0.0
    cand_idxs = _k_nearest_indices_by_deg_hist(generated_G, train_hists, n_bins, k)
    best_sim = -1.0
    best_G = None
    for idx in cand_idxs:
        G_tr = training_graphs_nx[idx]
        sim = wl_similarity_degree_only(generated_G, G_tr)
        if sim > best_sim:
            best_sim = sim
            best_G = G_tr
    if best_G is None:
        # fallback: compute against first element
        best_G = training_graphs_nx[0]
        best_sim = wl_similarity_degree_only(generated_G, best_G)
    return best_G, best_sim


def load_dataset(n_nodes):
    """Load dataset for specific node size."""
    data_path = f'data/labelhomophily0.5_{n_nodes}nodes_graphs.pkl'
    print(f"\nLoading dataset: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    
    print(f"Loaded {len(data_list)} graphs with {n_nodes} nodes each")
    return data_list, data_path


def train_autoencoder(data_list, run_name, output_dir):
    """Train VGAE (imported logic from main_comparison.py)."""
    from main_comparison import train_autoencoder as train_ae_original
    return train_ae_original(data_list, run_name, output_dir)


def train_denoiser(autoencoder, data_list, run_name, output_dir):
    """Train denoiser (imported logic from main_comparison.py)."""
    from main_comparison import train_denoiser as train_dn_original
    return train_dn_original(autoencoder, data_list, run_name, output_dir)


def generate_graphs(autoencoder, denoise_model, conditioning_stats, betas, num_samples=1):
    """Generate graphs (imported logic from main_comparison.py)."""
    from main_comparison import generate_graphs as gen_graphs_original
    return gen_graphs_original(autoencoder, denoise_model, conditioning_stats, betas, num_samples)


def run_experiment_for_n(n_nodes, split_config, output_dir):
    """
    Run experiment for specific graph complexity (n_nodes) with fixed N=1000.
    
    Args:
        n_nodes: Number of nodes per graph (complexity)
        split_config: Pre-split dataset pools
        output_dir: Output directory for this n
    
    Returns:
        Dictionary with metrics including WL similarities
    """
    N = FIXED_N  # Always use 1000 training graphs
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: n={n_nodes} nodes, N={N} training graphs (FIXED)")
    print(f"{'='*80}")
    
    # Create splits
    S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices = create_splits(split_config, N)
    
    # Create output directory
    exp_dir = output_dir / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple complexity ratio: n/N (graph complexity / dataset size)
    # For n=10, N=1000: 0.01 (simple task, lots of data → should generalize)
    # For n=1000, N=1000: 1.0 (complexity matches dataset → transition point)
    complexity_ratio = n_nodes / N if N > 0 else np.inf
    
    print(f"\nComplexity Metrics:")
    print(f"  Graph size (n): {n_nodes} nodes")
    print(f"  Dataset size (N): {N}")
    print(f"  Complexity ratio (n/N): {complexity_ratio:.4f}")
    
    # Distribution checks
    distribution_summary = perform_distribution_checks(S1, S2, exp_dir, N)
    
    # Train Gen1 (on S1)
    print(f"\n--- Training Gen1 on S1 subset ---")
    autoencoder_gen1, ae_gen1_metrics = train_autoencoder(S1, f"Gen1_N{N}", exp_dir)
    denoise_gen1, betas, dn_gen1_metrics = train_denoiser(autoencoder_gen1, S1, f"Gen1_N{N}", exp_dir)
    
    # Train Gen2 (on S2)
    print(f"\n--- Training Gen2 on S2 subset ---")
    autoencoder_gen2, ae_gen2_metrics = train_autoencoder(S2, f"Gen2_N{N}", exp_dir)
    denoise_gen2, _, dn_gen2_metrics = train_denoiser(autoencoder_gen2, S2, f"Gen2_N{N}", exp_dir)
    
    print(f"\nTraining Complete:")
    print(f"  Gen1 AE: {ae_gen1_metrics['best_val_loss']:.4f} | Gen1 DN: {dn_gen1_metrics['best_val_loss']:.4f}")
    print(f"  Gen2 AE: {ae_gen2_metrics['best_val_loss']:.4f} | Gen2 DN: {dn_gen2_metrics['best_val_loss']:.4f}")
    
    # Generate and evaluate
    print(f"\n--- Generating & Evaluating ---")
    generalization_scores = []  # Gen1 vs Gen2 (WL)
    gen_embed_sims = []         # Gen1 vs Gen2 (cosine in GIN embedding)
    # Robust memorization across all generated samples (symmetric): Gen1↔S1 and Gen2↔S2
    memorization_scores_gen1 = []    # Gen1 vs closest S1 training
    memorization_scores_gen2 = []    # Gen2 vs closest S2 training
    mem_embed_sims_gen1 = []         # Gen1 vs nearest S1 (cosine in embedding)
    mem_embed_sims_gen2 = []         # Gen2 vs nearest S2 (cosine in embedding)
    empty_gen1_count = 0
    empty_gen2_count = 0
    
    # Precompute S1 training graphs for memorization check
    S1_training_graphs_nx = []
    for data in S1:
        # Create adjacency matrix from edge_index if A doesn't exist
        if hasattr(data, 'A') and data.A is not None:
            adj = data.A[0].cpu().numpy()
        else:
            # Construct from edge_index (sparse representation)
            n_nodes = data.num_nodes
            adj = np.zeros((n_nodes, n_nodes))
            if data.edge_index.numel() > 0:
                edge_index = data.edge_index.cpu().numpy()
                adj[edge_index[0], edge_index[1]] = 1.0
        
        features = data.x.detach().cpu().numpy() if hasattr(data, 'x') else None
        G_train = construct_nx_from_adj(adj, node_features=features)
        S1_training_graphs_nx.append(G_train)
    
    print(f"Constructed {len(S1_training_graphs_nx)} training graphs for memorization check")
    print(f"Sample training graph: {S1_training_graphs_nx[0].number_of_nodes()} nodes, {S1_training_graphs_nx[0].number_of_edges()} edges")
    
    # Precompute S2 training graphs for closest-match visualization (mirror of S1)
    S2_training_graphs_nx = []
    for data in S2:
        if hasattr(data, 'A') and data.A is not None:
            adj = data.A[0].cpu().numpy()
        else:
            n_nodes = data.num_nodes
            adj = np.zeros((n_nodes, n_nodes))
            if data.edge_index.numel() > 0:
                edge_index = data.edge_index.cpu().numpy()
                adj[edge_index[0], edge_index[1]] = 1.0
        features = data.x.detach().cpu().numpy() if hasattr(data, 'x') else None
        G_train = construct_nx_from_adj(adj, node_features=features)
        S2_training_graphs_nx.append(G_train)

    # Precompute degree histograms for kNN prefilter
    n_bins = n_nodes  # degrees in [0, n-1]
    S1_degree_hists = _precompute_degree_hists(S1_training_graphs_nx, n_bins)
    S2_degree_hists = _precompute_degree_hists(S2_training_graphs_nx, n_bins)
    
    # Prepare to accumulate generated graphs for embedding-based metrics
    all_gen1_graphs = []
    all_gen2_graphs = []

    for i, test_graph in enumerate(tqdm(test_graphs, desc=f"Testing (n={n_nodes}, N={N})")):
        # Get conditioning stats
        if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) > i:
            c_test = test_stats_cache[i:i+1]
        else:
            c_test = test_graph.stats[:, :N_PROPERTIES]
        
        # Generate with SAME random seed for fair comparison
        torch.manual_seed(42 + i)
        G1_samples, _, _ = generate_graphs(autoencoder_gen1, denoise_gen1, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)
        
        torch.manual_seed(42 + i)  # Same seed for Gen2
        G2_samples, _, _ = generate_graphs(autoencoder_gen2, denoise_gen2, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)
        
        # Debug: Check first test case
        if i == 0:
            print(f"\nFirst test case debug:")
            print(f"  Gen1 samples: {[f'{G.number_of_nodes()}n/{G.number_of_edges()}e' for G in G1_samples]}")
            print(f"  Gen2 samples: {[f'{G.number_of_nodes()}n/{G.number_of_edges()}e' for G in G2_samples]}")
        
        # Compute generalization (Gen1 vs Gen2)
        for sample_idx in range(NUM_SAMPLES_PER_CONDITION):
            g1 = G1_samples[sample_idx]
            g2 = G2_samples[sample_idx]
            all_gen1_graphs.append(g1)
            all_gen2_graphs.append(g2)
            
            # Track empty graphs
            if g1.number_of_nodes() == 0:
                empty_gen1_count += 1
            if g2.number_of_nodes() == 0:
                empty_gen2_count += 1
            
            # Degree-only WL similarity between Gen1 and Gen2 samples
            sim = wl_similarity_degree_only(g1, g2)
            generalization_scores.append(sim)
            
            # Debug first few
            if i == 0 and sample_idx < 2:
                print(f"  Gen1 vs Gen2 sample {sample_idx}: WL similarity = {sim:.4f}")
        
        # Robust memorization with k-NN prefilter on degree histograms (WL)
        # Gen1 vs S1
        for g in G1_samples:
            _, sim_g1 = find_closest_graph_in_training_degree_only_k(g, S1_training_graphs_nx, S1_degree_hists, n_bins, K_NEAREST)
            memorization_scores_gen1.append(sim_g1)
        # Gen2 vs S2 (symmetric)
        for g in G2_samples:
            _, sim_g2 = find_closest_graph_in_training_degree_only_k(g, S2_training_graphs_nx, S2_degree_hists, n_bins, K_NEAREST)
            memorization_scores_gen2.append(sim_g2)
        
        # Debug: Print first few to check
        if i < 2:
            # Check node attributes safely
            gen_has_feat = False
            train_has_feat = False
            
            if G1_samples[0].number_of_nodes() > 0:
                first_node = list(G1_samples[0].nodes())[0]
                gen_has_feat = 'feature_vector' in G1_samples[0].nodes[first_node]
            # Print a quick nearest-neighbor sim snapshot for context
            dbg_closest, dbg_sim = find_closest_graph_in_training_degree_only_k(G1_samples[0], S1_training_graphs_nx, S1_degree_hists, n_bins, min(10, K_NEAREST))
            if dbg_closest is not None:
                print(f"  Test {i}: Gen1 sample0 ({G1_samples[0].number_of_nodes()}n, {G1_samples[0].number_of_edges()}e, has_feat={gen_has_feat}) "
                      f"vs closest S1 (deg-only WL), sim={dbg_sim:.4f}")
            # Manual degree-only WL test between first gen and first train
            if i == 0 and len(S1_training_graphs_nx) > 0:
                test_sim = wl_similarity_degree_only(G1_samples[0], S1_training_graphs_nx[0])
                print(f"  Direct WL test (degree-only) Gen1 vs first S1: {test_sim:.4f}")
    
    # Build untrained GIN embedding and compute cosine-based metrics
    embed_model = _RandomGIN(in_dim=2, hidden=64, out_dim=64, layers=3, seed=1234)
    # Embed training sets once
    E_S1 = _embed_graphs(embed_model, S1_training_graphs_nx)
    E_S2 = _embed_graphs(embed_model, S2_training_graphs_nx)
    # Embed all generated graphs
    E_G1 = _embed_graphs(embed_model, all_gen1_graphs)
    E_G2 = _embed_graphs(embed_model, all_gen2_graphs)

    # 1) Generalization (Gen1 vs Gen2) cosine for paired samples
    m = min(E_G1.shape[0], E_G2.shape[0])
    if m > 0:
        # Normalize then row-wise dot for pairs
        G1n = E_G1[:m] / (np.linalg.norm(E_G1[:m], axis=1, keepdims=True) + 1e-9)
        G2n = E_G2[:m] / (np.linalg.norm(E_G2[:m], axis=1, keepdims=True) + 1e-9)
        pair_cos = np.sum(G1n * G2n, axis=1)
        gen_embed_sims = pair_cos.tolist()

    # 2) Memorization (nearest training cosine in embedding space)
    # Gen1 -> S1
    if E_G1.shape[0] > 0 and E_S1.shape[0] > 0:
        C1 = _cosine_matrix(E_G1, E_S1)
        mem_embed_sims_gen1 = np.max(C1, axis=1).tolist()
    # Gen2 -> S2
    if E_G2.shape[0] > 0 and E_S2.shape[0] > 0:
        C2 = _cosine_matrix(E_G2, E_S2)
        mem_embed_sims_gen2 = np.max(C2, axis=1).tolist()

    gen_mean = np.mean(generalization_scores) if len(generalization_scores) > 0 else np.nan
    # Combine symmetric memorization scores for reporting/plots
    memorization_scores = memorization_scores_gen1 + memorization_scores_gen2
    mem_mean = np.mean(memorization_scores) if len(memorization_scores) > 0 else np.nan
    
    # Calculate empty graph statistics
    total_gen1_graphs = len(test_graphs) * NUM_SAMPLES_PER_CONDITION
    total_gen2_graphs = len(test_graphs) * NUM_SAMPLES_PER_CONDITION
    empty_gen1_pct = 100 * empty_gen1_count / total_gen1_graphs if total_gen1_graphs > 0 else 0
    empty_gen2_pct = 100 * empty_gen2_count / total_gen2_graphs if total_gen2_graphs > 0 else 0
    
    print(f"\nResults:")
    print(f"  Generalization (Gen1 vs Gen2): {gen_mean:.4f} ± {np.std(generalization_scores):.4f}")
    print(f"  Memorization (symmetric):      {mem_mean:.4f} ± {np.std(memorization_scores):.4f}")
    # Embedding metrics summary
    if len(gen_embed_sims) > 0:
        print(f"  [Emb] Gen1↔Gen2 cosine:       {np.mean(gen_embed_sims):.4f} ± {np.std(gen_embed_sims):.4f}")
    if len(mem_embed_sims_gen1) > 0:
        print(f"  [Emb] Gen1↔S1 (NN) cosine:   {np.mean(mem_embed_sims_gen1):.4f} ± {np.std(mem_embed_sims_gen1):.4f}")
    if len(mem_embed_sims_gen2) > 0:
        print(f"  [Emb] Gen2↔S2 (NN) cosine:   {np.mean(mem_embed_sims_gen2):.4f} ± {np.std(mem_embed_sims_gen2):.4f}")
    print(f"\n  [WARNING] Empty Graph Statistics:")
    print(f"    Gen1 empty: {empty_gen1_count}/{total_gen1_graphs} ({empty_gen1_pct:.1f}%)")
    print(f"    Gen2 empty: {empty_gen2_count}/{total_gen2_graphs} ({empty_gen2_pct:.1f}%)")
    if empty_gen1_pct > 10 or empty_gen2_pct > 10:
        print(f"    ⚠️  HIGH EMPTY RATE! Generated graphs are mostly empty - decoder not working properly!")
    
    # Save example graphs for visualization (first test case)
    print(f"\n--- Generating example graphs for visualization ---")
    torch.manual_seed(999)  # Fixed seed for consistent examples
    example_conditioning = test_stats_cache[0:1] if isinstance(test_stats_cache, torch.Tensor) else test_graphs[0].stats[:, :N_PROPERTIES]
    
    example_gen1, _, _ = generate_graphs(autoencoder_gen1, denoise_gen1, example_conditioning, betas, num_samples=3)
    torch.manual_seed(999)  # Same seed
    example_gen2, _, _ = generate_graphs(autoencoder_gen2, denoise_gen2, example_conditioning, betas, num_samples=3)
    
    # Find closest training matches for the example generated graphs
    closest_s1_for_gen1 = []
    closest_s1_sims = []
    for G in example_gen1:
        G_closest, sim = find_closest_graph_in_training_degree_only_k(G, S1_training_graphs_nx, S1_degree_hists, n_bins, K_NEAREST)
        closest_s1_for_gen1.append(G_closest)
        closest_s1_sims.append(sim)
    
    closest_s2_for_gen2 = []
    closest_s2_sims = []
    for G in example_gen2:
        G_closest, sim = find_closest_graph_in_training_degree_only_k(G, S2_training_graphs_nx, S2_degree_hists, n_bins, K_NEAREST)
        closest_s2_for_gen2.append(G_closest)
        closest_s2_sims.append(sim)
    
    # Example subsets from training (fallback/reference)
    # Fallback example slices (safe indexing)
    if len(S1_training_graphs_nx) > 0:
        idxs_s1 = [0, min(10, len(S1_training_graphs_nx)-1), min(20, len(S1_training_graphs_nx)-1)]
        example_s1 = [S1_training_graphs_nx[i] for i in idxs_s1]
    else:
        example_s1 = []
    if len(S2_training_graphs_nx) > 0:
        idxs_s2 = [0, min(10, len(S2_training_graphs_nx)-1), min(20, len(S2_training_graphs_nx)-1)]
        example_s2_graphs = [S2_training_graphs_nx[i] for i in idxs_s2]
    else:
        example_s2_graphs = []
    
    # Decide which memorization comparator to plot against Gen↔Gen for nicer separation
    chosen_mem_label = None
    chosen_mem_sims = None
    if len(gen_embed_sims) > 0:
        mean_gen = float(np.mean(gen_embed_sims))
        cand = []
        if len(mem_embed_sims_gen1) > 0:
            cand.append((abs(float(np.mean(mem_embed_sims_gen1)) - mean_gen), 'Gen1_vs_S1', mem_embed_sims_gen1))
        if len(mem_embed_sims_gen2) > 0:
            cand.append((abs(float(np.mean(mem_embed_sims_gen2)) - mean_gen), 'Gen2_vs_S2', mem_embed_sims_gen2))
        if len(cand) > 0:
            cand.sort(key=lambda x: x[0], reverse=True)
            _, chosen_mem_label, chosen_mem_sims = cand[0]

            # Plot simple histogram: Gen1↔Gen2 vs chosen memorization comparator (cosine)
            fig, ax = plt.subplots(figsize=(6, 5))
            bins = np.linspace(-1, 1, 41)
            ax.hist(gen_embed_sims, bins=bins, alpha=0.5, color='blue', density=True, edgecolor='black', label='Gen1 vs Gen2')
            ax.hist(chosen_mem_sims, bins=bins, alpha=0.5, color='orange', density=True, edgecolor='black', label=('Gen1 vs S1' if chosen_mem_label=='Gen1_vs_S1' else 'Gen2 vs S2'))
            ax.set_xlabel('Cosine similarity', fontsize=25)
            ax.set_ylabel('Density', fontsize=25)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlim(-1.0, 1.0)
            ax.legend(fontsize=12, loc='upper left')
            fig_dir = output_dir / 'figures'
            fig_dir.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_dir / f'embed_hist_mem_vs_gen_n{n_nodes}.png', dpi=300, bbox_inches='tight')
            plt.close()

    return {
        'n_nodes': n_nodes,
        'N': N,
        'exp_dir': exp_dir,
        'complexity_ratio': complexity_ratio,
        'ae_gen1_metrics': ae_gen1_metrics,
        'ae_gen2_metrics': ae_gen2_metrics,
        'dn_gen1_metrics': dn_gen1_metrics,
        'dn_gen2_metrics': dn_gen2_metrics,
        'generalization_scores': generalization_scores,
        'gen_embed_sims': gen_embed_sims,
        'mem_embed_sims_gen1': mem_embed_sims_gen1,
        'mem_embed_sims_gen2': mem_embed_sims_gen2,
        'chosen_mem_label': chosen_mem_label,
    # Distributions for plots and logs
    'memorization_scores': memorization_scores,
    'memorization_scores_gen1': memorization_scores_gen1,
    'memorization_scores_gen2': memorization_scores_gen2,
        'distribution_summary': distribution_summary,
        'example_gen1': example_gen1,
        'example_gen2': example_gen2,
        'example_s1': example_s1,
        'example_s2': example_s2_graphs,
        'closest_s1_for_gen1': closest_s1_for_gen1,
        'closest_s2_for_gen2': closest_s2_for_gen2,
        'closest_s1_sims': closest_s1_sims,
        'closest_s2_sims': closest_s2_sims,
    }


def visualize_example_graphs_single(result, n_nodes, output_dir):
    """
    Visualize example graphs for a single n value immediately after training.
    """
    print(f"  Creating 4-source example graphs for n={n_nodes}...")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Create 4x3 grid: 4 sources (Gen1, Closest S1, Gen2, Closest S2) x 3 examples each
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    
    sources = [
        ('Gen1', result['example_gen1'], '#3498db'),  # Blue
        ('Closest S1 to Gen1', result.get('closest_s1_for_gen1', result['example_s1']), '#2ecc71'),  # Green
        ('Gen2', result['example_gen2'], '#e74c3c'),  # Red
        ('Closest S2 to Gen2', result.get('closest_s2_for_gen2', result['example_s2']), '#f39c12'),  # Orange
    ]
    
    for row_idx, (label, graphs, color) in enumerate(sources):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(graphs):
                G = graphs[col_idx]
                
                # Layout
                if G.number_of_nodes() > 0:
                    try:
                        pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(G.number_of_nodes()))
                    except:
                        pos = nx.spring_layout(G, seed=42)
                    
                    # Draw
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, 
                                          node_size=200, alpha=0.8, edgecolors='black', linewidths=1)
                    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                          width=1.5, alpha=0.6)
                    
                    # Info
                    info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                    ax.text(0.5, -0.15, info_text, transform=ax.transAxes,
                           ha='center', fontsize=11, fontweight='bold')
            
            ax.set_axis_off()
            
            # Labels
            if col_idx == 0:
                ax.text(-0.3, 0.5, label, transform=ax.transAxes, 
                       fontsize=14, fontweight='bold', rotation=90, 
                       va='center', ha='center', color=color)
            
            if row_idx == 0:
                ax.set_title(f'Example {col_idx+1}', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Generated vs Closest Training Matches: n={n_nodes} nodes (N={FIXED_N} training)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    plt.savefig(fig_dir / f'gen_vs_closest_train_n{n_nodes}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_dir / f'gen_vs_closest_train_n{n_nodes}.png'}")
    plt.close()


def visualize_example_graphs(all_results, output_dir):
    """
    Visualize example graphs from Gen1, Gen2, S1, and S2 for each node size.
    Shows visual evidence of memorization vs generalization.
    """
    print(f"\n--- Creating example graph visualizations ---")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    for n_nodes in NODE_SIZES:
        if n_nodes not in all_results:
            continue
        
        result = all_results[n_nodes]
        
        # Create 4x3 grid: 4 sources (Gen1, Closest S1, Gen2, Closest S2) x 3 examples each
        fig, axes = plt.subplots(4, 3, figsize=(15, 18))
        
        sources = [
            ('Gen1', result['example_gen1'], '#3498db'),  # Blue
            ('Closest S1 to Gen1', result.get('closest_s1_for_gen1', result['example_s1']), '#2ecc71'),  # Green
            ('Gen2', result['example_gen2'], '#e74c3c'),  # Red
            ('Closest S2 to Gen2', result.get('closest_s2_for_gen2', result['example_s2']), '#f39c12'),  # Orange
        ]
        
        for row_idx, (label, graphs, color) in enumerate(sources):
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                
                if col_idx < len(graphs):
                    G = graphs[col_idx]
                    
                    # Layout
                    if G.number_of_nodes() > 0:
                        try:
                            pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(G.number_of_nodes()))
                        except:
                            pos = nx.spring_layout(G, seed=42)
                        
                        # Draw
                        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, 
                                              node_size=200, alpha=0.8, edgecolors='black', linewidths=1)
                        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                              width=1.5, alpha=0.6)
                        
                        # Info
                        info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                        ax.text(0.5, -0.15, info_text, transform=ax.transAxes,
                               ha='center', fontsize=11, fontweight='bold')
                
                ax.set_axis_off()
                
                # Labels
                if col_idx == 0:
                    ax.text(-0.3, 0.5, label, transform=ax.transAxes, 
                           fontsize=14, fontweight='bold', rotation=90, 
                           va='center', ha='center', color=color)
                
                if row_idx == 0:
                    ax.set_title(f'Example {col_idx+1}', fontsize=12, fontweight='bold')
        
    plt.suptitle(f'Generated vs Closest Training Matches: n={n_nodes} nodes (N={FIXED_N} training)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    plt.savefig(fig_dir / f'gen_vs_closest_train_n{n_nodes}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / f'gen_vs_closest_train_n{n_nodes}.png'}")
    plt.close()


def visualize_gt_vs_generated_single(result, n_nodes, output_dir):
    """
    Create GT vs Generated visualization for a single n value immediately after training.
    """
    print(f"  Creating GT vs Generated for n={n_nodes}...")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Create 2 rows x 6 columns: Top row = GT (S1), Bottom row = Generated (Gen1)
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    # Get GT and Generated graphs
    gt_graphs = result['example_s1'][:6]  # Up to 6 GT examples
    gen_graphs = result['example_gen1'][:6]  # Up to 6 Generated examples
    
    # Top row: Ground Truth (Training Data)
    for col_idx in range(6):
        ax = axes[0, col_idx]
        
        if col_idx < len(gt_graphs):
            G = gt_graphs[col_idx]
            
            if G.number_of_nodes() > 0:
                try:
                    pos = nx.spring_layout(G, seed=42+col_idx, k=1/np.sqrt(G.number_of_nodes()))
                except:
                    pos = nx.spring_layout(G, seed=42+col_idx)
                
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#2ecc71', 
                                      node_size=300, alpha=0.8, edgecolors='black', linewidths=1.5)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                      width=2, alpha=0.7)
                
                info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                       ha='center', fontsize=14, fontweight='bold')
        
        ax.set_axis_off()
        if col_idx == 0:
            ax.text(-0.2, 0.5, 'Ground Truth\n(Training)', transform=ax.transAxes,
                   ha='right', va='center', fontsize=16, fontweight='bold', color='#2ecc71')
    
    # Bottom row: Generated graphs
    for col_idx in range(6):
        ax = axes[1, col_idx]
        
        if col_idx < len(gen_graphs):
            G = gen_graphs[col_idx]
            
            if G.number_of_nodes() > 0:
                try:
                    pos = nx.spring_layout(G, seed=42+col_idx, k=1/np.sqrt(G.number_of_nodes()))
                except:
                    pos = nx.spring_layout(G, seed=42+col_idx)
                
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#3498db', 
                                      node_size=300, alpha=0.8, edgecolors='black', linewidths=1.5)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                      width=2, alpha=0.7)
                
                info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                       ha='center', fontsize=14, fontweight='bold')
        
        ax.set_axis_off()
        if col_idx == 0:
            ax.text(-0.2, 0.5, 'Generated\n(Gen1)', transform=ax.transAxes,
                   ha='right', va='center', fontsize=16, fontweight='bold', color='#3498db')
    
    fig.suptitle(f'Ground Truth vs Generated Graphs (n={n_nodes} nodes, N={FIXED_N} training examples)', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(fig_dir / f'gt_vs_generated_n{n_nodes}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_dir / f'gt_vs_generated_n{n_nodes}.png'}")
    plt.close()


def visualize_gt_vs_generated(all_results, output_dir):
    """
    Create side-by-side comparison of Ground Truth (training) vs Generated graphs for each n.
    This helps understand if models are truly memorizing or learning the distribution.
    """
    print(f"\n--- Creating GT vs Generated comparison visualizations ---")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    for n_nodes in NODE_SIZES:
        if n_nodes not in all_results:
            continue
        
        result = all_results[n_nodes]
        
        # Create 2 rows x 6 columns: Top row = GT (S1), Bottom row = Generated (Gen1)
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # Get GT and Generated graphs
        gt_graphs = result['example_s1'][:6]  # Up to 6 GT examples
        gen_graphs = result['example_gen1'][:6]  # Up to 6 Generated examples
        
        # Top row: Ground Truth (Training Data)
        for col_idx in range(6):
            ax = axes[0, col_idx]
            
            if col_idx < len(gt_graphs):
                G = gt_graphs[col_idx]
                
                if G.number_of_nodes() > 0:
                    try:
                        pos = nx.spring_layout(G, seed=42+col_idx, k=1/np.sqrt(G.number_of_nodes()))
                    except:
                        pos = nx.spring_layout(G, seed=42+col_idx)
                    
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#2ecc71', 
                                          node_size=300, alpha=0.8, edgecolors='black', linewidths=1.5)
                    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                          width=2, alpha=0.7)
                    
                    info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                    ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                           ha='center', fontsize=14, fontweight='bold')
            
            ax.set_axis_off()
            if col_idx == 0:
                ax.text(-0.2, 0.5, 'Ground Truth\n(Training)', transform=ax.transAxes,
                       ha='right', va='center', fontsize=16, fontweight='bold', color='#2ecc71')
        
        # Bottom row: Generated graphs
        for col_idx in range(6):
            ax = axes[1, col_idx]
            
            if col_idx < len(gen_graphs):
                G = gen_graphs[col_idx]
                
                if G.number_of_nodes() > 0:
                    try:
                        pos = nx.spring_layout(G, seed=42+col_idx, k=1/np.sqrt(G.number_of_nodes()))
                    except:
                        pos = nx.spring_layout(G, seed=42+col_idx)
                    
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#3498db', 
                                          node_size=300, alpha=0.8, edgecolors='black', linewidths=1.5)
                    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                          width=2, alpha=0.7)
                    
                    info_text = f"{G.number_of_nodes()}n, {G.number_of_edges()}e"
                    ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                           ha='center', fontsize=14, fontweight='bold')
            
            ax.set_axis_off()
            if col_idx == 0:
                ax.text(-0.2, 0.5, 'Generated\n(Gen1)', transform=ax.transAxes,
                       ha='right', va='center', fontsize=16, fontweight='bold', color='#3498db')
        
        fig.suptitle(f'Ground Truth vs Generated Graphs (n={n_nodes} nodes, N={FIXED_N} training examples)', 
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0.05, 0, 1, 0.96])
        plt.savefig(fig_dir / f'gt_vs_generated_n{n_nodes}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_dir / f'gt_vs_generated_n{n_nodes}.png'}")
        plt.close()


def create_aggregate_visualizations(all_results, output_dir):
    """
    Create aggregate visualizations showing memorization→generalization across node sizes.
    With N=1000 fixed, we show how graph complexity (n) affects the transition.
    """
    print(f"\n--- Creating aggregate visualizations ---")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Extract data for all n values
    n_vals = []
    gen_means = []
    gen_stds = []
    mem_means = []
    mem_stds = []
    complexity_ratios = []
    
    for n_nodes in NODE_SIZES:
        if n_nodes in all_results:
            result = all_results[n_nodes]
            n_vals.append(n_nodes)
            
            gen_scores = result['generalization_scores']
            mem_scores = result['memorization_scores']
            
            gen_means.append(np.mean(gen_scores))
            gen_stds.append(np.std(gen_scores))
            mem_means.append(np.mean(mem_scores))
            mem_stds.append(np.std(mem_scores))
            complexity_ratios.append(result['complexity_ratio'])
    
    # 1. Main histogram plot: Memorization to Generalization as n increases (histograms, no KDE)
    n_plots = len(n_vals)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, n_nodes in enumerate(n_vals):
        result = all_results[n_nodes]
        ax = axes[idx]
        
        gen_scores = result['generalization_scores']
        mem_scores = result['memorization_scores']
        
        # Plot overlapping histograms (normalized) instead of KDE
        bins = np.linspace(0, 1, 21)
        if len(mem_scores) > 0:
            ax.hist(mem_scores, bins=bins, alpha=0.5, color='orange', label='Memorization', density=True, edgecolor='black')
        if len(gen_scores) > 0:
            ax.hist(gen_scores, bins=bins, alpha=0.5, color='blue', label='Generalization', density=True, edgecolor='black')
        
        # Add mean lines
        gen_mean = np.mean(gen_scores)
        mem_mean = np.mean(mem_scores)
        ax.axvline(gen_mean, color='darkblue', linestyle='--', linewidth=2.5, alpha=0.9)
        ax.axvline(mem_mean, color='darkorange', linestyle='--', linewidth=2.5, alpha=0.9)
        
        # Add text annotations showing means
        ax.text(gen_mean, ax.get_ylim()[1]*0.9, f'{gen_mean:.2f}', 
                ha='center', fontsize=12, color='darkblue', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(mem_mean, ax.get_ylim()[1]*0.8, f'{mem_mean:.2f}', 
                ha='center', fontsize=12, color='darkorange', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('WL Similarity', fontsize=25)
        ax.set_ylabel('Density', fontsize=25)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=14)
        if idx == 0:
            ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(fig_dir / 'memorization_to_generalization_histograms.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'memorization_to_generalization_histograms.png'}")
    plt.close()
    
    # 2. Convergence line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(n_vals, gen_means, yerr=gen_stds, marker='o', color='blue',
                linewidth=2, markersize=8, capsize=5, label='Generalization (Gen1 vs Gen2)')
    ax.errorbar(n_vals, mem_means, yerr=mem_stds, marker='s', color='orange',
                linewidth=2, markersize=8, capsize=5, label='Memorization (vs training)')
    ax.set_xlabel('Graph Complexity (number of nodes)', fontsize=25)
    ax.set_ylabel('WL Kernel Similarity', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'wl_similarity_vs_complexity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'wl_similarity_vs_complexity.png'}")
    plt.close()
    
    # 3. Memorization vs Generalization Difference Plot (KEY METRIC)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the difference: Mem - Gen
    # Positive = Memorization regime, Negative = Generalization regime
    mem_gen_diff = np.array(mem_means) - np.array(gen_means)
    
    colors = ['red' if diff > 0 else 'green' for diff in mem_gen_diff]
    ax.bar(range(len(n_vals)), mem_gen_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add shaded regions
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='red', label='Memorization Regime')
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='green', label='Generalization Regime')
    
    ax.set_xlabel('Graph Complexity (number of nodes)', fontsize=25)
    ax.set_ylabel('Mem_WL - Gen_WL', fontsize=25)
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels([f'n={n}' for n in n_vals], fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / 'memorization_vs_generalization_difference.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'memorization_vs_generalization_difference.png'}")
    plt.close()
    
    # 4. Complexity ratio plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_vals, complexity_ratios, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax.set_xlabel('Graph Complexity (number of nodes)', fontsize=25)
    ax.set_ylabel('Complexity/Dataset Ratio', fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'complexity_ratio_vs_n.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'complexity_ratio_vs_n.png'}")
    plt.close()


def save_summary(all_results, output_dir):
    """Save comprehensive summary across all n values (N=1000 fixed)."""
    summary_path = output_dir / "experiment_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Graph Complexity Study: Memorization to Generalization\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fixed training size: N = {FIXED_N} graphs\n")
        f.write(f"Node sizes tested: {NODE_SIZES}\n\n")
        
        f.write("Results:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'n_nodes':<10} {'ComplexRatio':<14} {'Gen_WL_Mean':<14} {'Gen_WL_Std':<14} "
                f"{'Mem_WL_Mean':<14} {'Mem_WL_Std':<14}\n")
        f.write("-"*100 + "\n")
        
        for n_nodes in NODE_SIZES:
            if n_nodes in all_results:
                r = all_results[n_nodes]
                gen_mean = np.mean(r['generalization_scores'])
                gen_std = np.std(r['generalization_scores'])
                mem_mean = np.mean(r['memorization_scores'])
                mem_std = np.std(r['memorization_scores'])
                
                f.write(f"{n_nodes:<10} {r['complexity_ratio']:<14.4f} {gen_mean:<14.4f} {gen_std:<14.4f} "
                       f"{mem_mean:<14.4f} {mem_std:<14.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nInterpretation:\n")
        f.write("-"*80 + "\n")
        f.write("As graph complexity (n) increases:\n")
        f.write("- Small n (simple graphs): Models should generalize (Gen_WL ≈ Mem_WL)\n")
        f.write("- Large n (complex graphs): Models may memorize (Gen_WL < Mem_WL)\n")
        f.write("- Transition point shows where complexity overwhelms training data\n")
        f.write("\nWith fixed N=1000, we observe how task complexity affects learning:\n")
        f.write("- High complexity/dataset ratio → memorization regime\n")
        f.write("- Low complexity/dataset ratio → generalization regime\n")
    
    print(f"\nSaved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Graph Complexity Study: Memorization to Generalization')
    parser.add_argument('--output-dir', type=str, default='outputs/nodesize_study',
                       help='Output directory')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom run name')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: fewer epochs/timesteps and samples for a quick smoke test')
    parser.add_argument('--fixed-n', type=int, default=None,
                       help='Override FIXED_N training set size')
    parser.add_argument('--epochs-ae', type=int, default=None,
                       help='Override autoencoder epochs')
    parser.add_argument('--epochs-dn', type=int, default=None,
                       help='Override denoiser epochs')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override diffusion timesteps')
    parser.add_argument('--node-sizes', type=str, default=None,
                       help='Comma-separated list of node sizes to run (e.g., "10,30,50")')
    parser.add_argument('--k-nearest', type=int, default=100,
                       help='Use k-nearest shortlist for training matching (degree-hist prefilter)')
    args = parser.parse_args()

    # Optionally override module-level settings for quick experimentation
    global FIXED_N, NODE_SIZES, EPOCHS_AUTOENCODER, EPOCHS_DENOISER, TIMESTEPS, NUM_SAMPLES_PER_CONDITION, K_NEAREST
    if args.fixed_n is not None:
        FIXED_N = args.fixed_n
    if args.node_sizes is not None:
        try:
            NODE_SIZES = [int(x) for x in args.node_sizes.split(',') if x.strip()]
        except Exception:
            print(f"Warning: failed to parse --node-sizes='{args.node_sizes}', using default {NODE_SIZES}")
    if args.epochs_ae is not None:
        EPOCHS_AUTOENCODER = args.epochs_ae
    if args.epochs_dn is not None:
        EPOCHS_DENOISER = args.epochs_dn
    if args.timesteps is not None:
        TIMESTEPS = args.timesteps
    if args.fast:
        # Conservative fast-mode defaults for a quick smoke test
        EPOCHS_AUTOENCODER = min(EPOCHS_AUTOENCODER, 5)
        EPOCHS_DENOISER = min(EPOCHS_DENOISER, 5)
        TIMESTEPS = min(TIMESTEPS, 100)
        NUM_SAMPLES_PER_CONDITION = min(NUM_SAMPLES_PER_CONDITION, 2)
        K_NEAREST = min(K_NEAREST, 50)
    if args.k_nearest is not None and args.k_nearest > 0:
        K_NEAREST = args.k_nearest
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{args.run_name}_{timestamp}"
    else:
        run_id = f"nodesize_{timestamp}"
    
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize incremental metrics progress log
    progress_path = output_dir / "metrics_progress.txt"
    with open(progress_path, 'w') as pf:
        pf.write("n_nodes\tGen_WL_Mean\tGen_WL_Std\tMem_WL_Mean\tMem_WL_Std\n")
    
    print(f"\n{'='*80}")
    print("Graph Complexity Study: Memorization to Generalization")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Fixed training size: N = {FIXED_N}")
    print(f"Node sizes: {NODE_SIZES}")
    if args.fast:
        print(f"Fast mode: AE epochs={EPOCHS_AUTOENCODER}, DN epochs={EPOCHS_DENOISER}, timesteps={TIMESTEPS}, samples/cond={NUM_SAMPLES_PER_CONDITION}")
    print(f"k-nearest shortlist size: K={K_NEAREST}")
    print(f"{'='*80}\n")
    
    # Run experiments for each node size
    all_results = {}
    
    for n_nodes in NODE_SIZES:
        print(f"\n{'#'*80}")
        print(f"# GRAPH COMPLEXITY: n = {n_nodes} nodes")
        print(f"{'#'*80}")
        
        # Load dataset for this node size
        try:
            data_list, data_path = load_dataset(n_nodes)
        except FileNotFoundError as e:
            print(f"Skipping n={n_nodes}: {e}")
            continue
        
        # Create output dir for this n
        n_output_dir = output_dir / f"n_{n_nodes}"
        n_output_dir.mkdir(exist_ok=True)
        
        # Split dataset
        cache_path = Path(data_path).parent / "cache" / f"labelhomophily0.5_{n_nodes}nodes_test_stats_seed{SPLIT_SEED}_size{TEST_SET_SIZE}.pt"
        split_config = shuffle_and_split_dataset(
            data_list,
            test_size=TEST_SET_SIZE,
            seed=SPLIT_SEED,
            stats_cache_path=cache_path
        )
        
        # Run experiment with fixed N=1000
        result = run_experiment_for_n(n_nodes, split_config, n_output_dir)
        all_results[n_nodes] = result
        
        # Save results
        with open(n_output_dir / f'results_n{n_nodes}.pkl', 'wb') as f:
            pickle.dump(result, f)
        
        # Append WL metrics for this n to progress log for early inspection
        gen_mean = float(np.mean(result['generalization_scores'])) if len(result['generalization_scores']) > 0 else float('nan')
        gen_std = float(np.std(result['generalization_scores'])) if len(result['generalization_scores']) > 0 else float('nan')
        mem_mean = float(np.mean(result['memorization_scores'])) if len(result['memorization_scores']) > 0 else float('nan')
        mem_std = float(np.std(result['memorization_scores'])) if len(result['memorization_scores']) > 0 else float('nan')
        with open(progress_path, 'a') as pf:
            pf.write(f"{n_nodes}\t{gen_mean:.4f}\t{gen_std:.4f}\t{mem_mean:.4f}\t{mem_std:.4f}\n")
        
        # Create immediate visualization for this n (don't wait till end)
        print(f"\n--- Creating immediate visualizations for n={n_nodes} ---")
        visualize_gt_vs_generated_single(result, n_nodes, output_dir)
        visualize_example_graphs_single(result, n_nodes, output_dir)
    
    # Create visualizations
    visualize_gt_vs_generated(all_results, output_dir)  # GT vs Generated comparison
    visualize_example_graphs(all_results, output_dir)   # 4-source comparison
    create_aggregate_visualizations(all_results, output_dir)  # Aggregate plots
    
    # Save overall summary
    save_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Main histogram (Mem vs Gen): {output_dir}/figures/memorization_to_generalization_histograms.png")
    print(f"  - Mem vs Gen difference: {output_dir}/figures/memorization_vs_generalization_difference.png")
    print(f"  - WL similarity vs n: {output_dir}/figures/wl_similarity_vs_complexity.png")
    print(f"  - Complexity ratio: {output_dir}/figures/complexity_ratio_vs_n.png")
    print(f"  - GT vs Generated: {output_dir}/figures/gt_vs_generated_n*.png (for each n)")
    print(f"  - Gen vs Closest Train: {output_dir}/figures/gen_vs_closest_train_n*.png (for each n)")
    print(f"\nInterpretation:")
    print(f"  - As n increases (left to right in histogram), observe transition")
    print(f"  - Small n: Task is easy → both Mem and Gen high (model learns quickly)")
    print(f"  - Large n: Complexity increases → forced to generalize")
    print(f"  - Fixed N={FIXED_N} shows how complexity affects learning")
    print(f"\nGraph visualizations show:")
    print(f"  - GT vs Generated: Side-by-side comparison of training vs generated graphs")
    print(f"    → For small n: Should look similar (easy task)")
    print(f"    → For large n: More variation (complex task)")
    print(f"  - 4-source comparison: Gen1/Gen2/S1/S2 examples")
    print(f"    → Gen1 (blue) vs Gen2 (red): Do they look similar? (generalization)")
    print(f"    → Gen1 vs S1 (green): Does Gen1 copy S1? (memorization)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
