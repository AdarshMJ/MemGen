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

from torch_geometric.loader import DataLoader
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
FIXED_N = 1000  # Fixed training set size - always use 1000 graphs from S1 and S2
NODE_SIZES = [5, 10, 20, 30, 40, 50, 100,500]  # Graph complexities to test (loop over n)
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
TIMESTEPS = 500
NUM_SAMPLES_PER_CONDITION = 5

BETA_KL_WEIGHT = 0.05
SMALL_DATASET_THRESHOLD = 50
SMALL_DATASET_KL_WEIGHT = 0.01
SMALL_DATASET_DROPOUT = 0.1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    generalization_scores = []  # Gen1 vs Gen2
    memorization_scores = []    # Gen1 vs S1 training
    
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
            sim = compute_wl_similarity(G1_samples[sample_idx], G2_samples[sample_idx])
            generalization_scores.append(sim)
            
            # Debug first few
            if i == 0 and sample_idx < 2:
                print(f"  Gen1 vs Gen2 sample {sample_idx}: WL similarity = {sim:.4f}")
        
        # Compute memorization (Gen1 vs closest S1 training graph)
        closest_G, sim = find_closest_graph_in_training(G1_samples[0], S1_training_graphs_nx)
        memorization_scores.append(sim)
        
        # Debug: Print first few to check
        if i < 2:
            # Check node attributes safely
            gen_has_feat = False
            train_has_feat = False
            
            if G1_samples[0].number_of_nodes() > 0:
                first_node = list(G1_samples[0].nodes())[0]
                gen_has_feat = 'feature_vector' in G1_samples[0].nodes[first_node]
            
            if closest_G.number_of_nodes() > 0:
                first_node = list(closest_G.nodes())[0]
                train_has_feat = 'feature_vector' in closest_G.nodes[first_node]
            
            print(f"  Test {i}: Generated graph ({G1_samples[0].number_of_nodes()}n, {G1_samples[0].number_of_edges()}e, has_feat={gen_has_feat}) "
                  f"vs closest training ({closest_G.number_of_nodes()}n, {closest_G.number_of_edges()}e, has_feat={train_has_feat}), sim={sim:.4f}")
            
            # Manual test of WL similarity
            if i == 0:
                test_sim = compute_wl_similarity(G1_samples[0], S1_training_graphs_nx[0])
                print(f"  Direct WL test (Gen1 vs first training): {test_sim:.4f}")
    
    gen_mean = np.mean(generalization_scores) if len(generalization_scores) > 0 else np.nan
    mem_mean = np.mean(memorization_scores) if len(memorization_scores) > 0 else np.nan
    
    print(f"\nResults:")
    print(f"  Generalization (Gen1 vs Gen2): {gen_mean:.4f} ± {np.std(generalization_scores):.4f}")
    print(f"  Memorization (Gen1 vs S1):     {mem_mean:.4f} ± {np.std(memorization_scores):.4f}")
    
    # Save example graphs for visualization (first test case)
    print(f"\n--- Generating example graphs for visualization ---")
    torch.manual_seed(999)  # Fixed seed for consistent examples
    example_conditioning = test_stats_cache[0:1] if isinstance(test_stats_cache, torch.Tensor) else test_graphs[0].stats[:, :N_PROPERTIES]
    
    example_gen1, _, _ = generate_graphs(autoencoder_gen1, denoise_gen1, example_conditioning, betas, num_samples=3)
    torch.manual_seed(999)  # Same seed
    example_gen2, _, _ = generate_graphs(autoencoder_gen2, denoise_gen2, example_conditioning, betas, num_samples=3)
    
    # Get example training graphs from S1 and S2
    example_s1 = [S1_training_graphs_nx[i] for i in [0, 10, 20]]
    example_s2_graphs = []
    for data in [S2[0], S2[10], S2[20]]:
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
        example_s2_graphs.append(construct_nx_from_adj(adj, node_features=features))
    
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
        'memorization_scores': memorization_scores,
        'distribution_summary': distribution_summary,
        'example_gen1': example_gen1,
        'example_gen2': example_gen2,
        'example_s1': example_s1,
        'example_s2': example_s2_graphs,
    }


def visualize_example_graphs_single(result, n_nodes, output_dir):
    """
    Visualize example graphs for a single n value immediately after training.
    """
    print(f"  Creating 4-source example graphs for n={n_nodes}...")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Create 4x3 grid: 4 sources (Gen1, Gen2, S1, S2) x 3 examples each
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    
    sources = [
        ('Gen1', result['example_gen1'], '#3498db'),  # Blue
        ('Gen2', result['example_gen2'], '#e74c3c'),  # Red
        ('S1 (Training)', result['example_s1'], '#2ecc71'),  # Green
        ('S2 (Training)', result['example_s2'], '#f39c12'),  # Orange
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
    
    plt.suptitle(f'Graph Examples: n={n_nodes} nodes (N={FIXED_N} training)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    plt.savefig(fig_dir / f'example_graphs_n{n_nodes}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_dir / f'example_graphs_n{n_nodes}.png'}")
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
        
        # Create 4x3 grid: 4 sources (Gen1, Gen2, S1, S2) x 3 examples each
        fig, axes = plt.subplots(4, 3, figsize=(15, 18))
        
        sources = [
            ('Gen1', result['example_gen1'], '#3498db'),  # Blue
            ('Gen2', result['example_gen2'], '#e74c3c'),  # Red
            ('S1 (Training)', result['example_s1'], '#2ecc71'),  # Green
            ('S2 (Training)', result['example_s2'], '#f39c12'),  # Orange
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
        
        plt.suptitle(f'Graph Examples: n={n_nodes} nodes (N={FIXED_N} training)', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0.05, 0, 1, 0.99])
        plt.savefig(fig_dir / f'example_graphs_n{n_nodes}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_dir / f'example_graphs_n{n_nodes}.png'}")
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
    
    # 1. Main histogram plot: Memorization to Generalization as n increases
    n_plots = len(n_vals)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, n_nodes in enumerate(n_vals):
        result = all_results[n_nodes]
        ax = axes[idx]
        
        gen_scores = result['generalization_scores']
        mem_scores = result['memorization_scores']
        
        # Use kernel density estimation for smoother visualization
        from scipy.stats import gaussian_kde
        
        # Create KDE for both distributions
        x_range = np.linspace(0, 1, 200)
        
        if len(mem_scores) > 1:
            kde_mem = gaussian_kde(mem_scores, bw_method=0.1)
            density_mem = kde_mem(x_range)
            ax.fill_between(x_range, density_mem, alpha=0.5, color='orange', label='Memorization')
            ax.plot(x_range, density_mem, color='darkorange', linewidth=2)
        
        if len(gen_scores) > 1:
            kde_gen = gaussian_kde(gen_scores, bw_method=0.1)
            density_gen = kde_gen(x_range)
            ax.fill_between(x_range, density_gen, alpha=0.5, color='blue', label='Generalization')
            ax.plot(x_range, density_gen, color='darkblue', linewidth=2)
        
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
        
        ax.set_xlabel('WL Similarity', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(f'n={n_nodes}', fontsize=18, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=14)
        if idx == 0:
            ax.legend(fontsize=12, loc='upper left')
    
    fig.suptitle(f'Memorization to Generalization: Increasing Graph Complexity (N={FIXED_N} fixed)', 
                 fontsize=20, y=1.02)
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
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{args.run_name}_{timestamp}"
    else:
        run_id = f"nodesize_{timestamp}"
    
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Graph Complexity Study: Memorization to Generalization")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Fixed training size: N = {FIXED_N}")
    print(f"Node sizes: {NODE_SIZES}")
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
    print(f"  - Main histogram (KDE): {output_dir}/figures/memorization_to_generalization_histograms.png")
    print(f"  - Mem vs Gen difference: {output_dir}/figures/memorization_vs_generalization_difference.png")
    print(f"  - WL similarity vs n: {output_dir}/figures/wl_similarity_vs_complexity.png")
    print(f"  - Complexity ratio: {output_dir}/figures/complexity_ratio_vs_n.png")
    print(f"  - GT vs Generated: {output_dir}/figures/gt_vs_generated_n*.png (for each n)")
    print(f"  - Example 4-source: {output_dir}/figures/example_graphs_n*.png (for each n)")
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
