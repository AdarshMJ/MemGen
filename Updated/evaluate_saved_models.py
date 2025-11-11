#!/usr/bin/env python3
"""
Evaluate saved models without retraining.

Loads pre-trained autoencoder and denoiser checkpoints, generates graphs,
and produces evaluation metrics and visualizations using WL degree-only matching.

Usage:
    python evaluate_saved_models.py --checkpoint-dir outputs/nodesize_study/LatestWorking --node-sizes 5 10 20 30 50
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Import from main_nodesize
from main_nodesize import (
    device, N_MAX_NODES, N_PROPERTIES, LATENT_DIM, HIDDEN_DIM_ENCODER,
    HIDDEN_DIM_DECODER, HIDDEN_DIM_DENOISE, TIMESTEPS, USE_BIAS,
    load_dataset, create_splits, linear_beta_schedule, generate_graphs,
    wl_similarity_degree_only,
    construct_nx_from_adj, _precompute_degree_hists_from_pyg,
    find_closest_graph_in_training_degree_only_k,
    _pyg_to_nx
)
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN


def visualize_gt_vs_generated(s1_graphs, s2_graphs, gen1_graphs, gen2_graphs, n_nodes, output_dir):
    """Visualize ground truth vs generated graphs."""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(f'Ground Truth (S1/S2) vs Generated (Gen1/Gen2) - n={n_nodes}', fontsize=16)
    
    # Adaptive node size and edge width based on graph size
    if n_nodes <= 50:
        node_size = 100
        edge_width = 1.0
        layout_k = None  # Use default
    elif n_nodes <= 100:
        node_size = 50
        edge_width = 0.5
        layout_k = None
    elif n_nodes <= 300:
        node_size = 10
        edge_width = 0.2
        layout_k = 0.3
    else:  # Large graphs (600+)
        node_size = 2
        edge_width = 0.1
        layout_k = 0.5
    
    # Plot S1 samples
    for i in range(3):
        ax = axes[0, i]
        if i < len(s1_graphs) and s1_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(s1_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(s1_graphs[i], pos, ax=ax, node_size=node_size, node_color='lightblue', 
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'S1 #{i+1} ({s1_graphs[i].number_of_nodes()}n/{s1_graphs[i].number_of_edges()}e)')
        ax.axis('off')
    
    # Plot S2 samples
    for i in range(3):
        ax = axes[1, i]
        if i < len(s2_graphs) and s2_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(s2_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(s2_graphs[i], pos, ax=ax, node_size=node_size, node_color='lightcoral',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'S2 #{i+1} ({s2_graphs[i].number_of_nodes()}n/{s2_graphs[i].number_of_edges()}e)')
        ax.axis('off')
    
    # Plot Gen1 samples
    for i in range(3):
        ax = axes[0, i+3]
        if i < len(gen1_graphs) and gen1_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(gen1_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(gen1_graphs[i], pos, ax=ax, node_size=node_size, node_color='cyan',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'Gen1 #{i+1} ({gen1_graphs[i].number_of_nodes()}n/{gen1_graphs[i].number_of_edges()}e)')
        ax.axis('off')
    
    # Plot Gen2 samples
    for i in range(3):
        ax = axes[1, i+3]
        if i < len(gen2_graphs) and gen2_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(gen2_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(gen2_graphs[i], pos, ax=ax, node_size=node_size, node_color='salmon',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'Gen2 #{i+1} ({gen2_graphs[i].number_of_nodes()}n/{gen2_graphs[i].number_of_edges()}e)')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = output_dir / f'gt_vs_generated_n{n_nodes}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def visualize_gen_vs_closest_train(gen1_graphs, gen2_graphs, closest_s1, closest_s2, 
                                    sims_s1, sims_s2, n_nodes, output_dir):
    """Visualize generated graphs vs their closest training matches."""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(f'Generated vs Closest Training Match - n={n_nodes}', fontsize=16)
    
    # Adaptive node size and edge width based on graph size
    if n_nodes <= 50:
        node_size = 100
        edge_width = 1.0
        layout_k = None
    elif n_nodes <= 100:
        node_size = 50
        edge_width = 0.5
        layout_k = None
    elif n_nodes <= 300:
        node_size = 10
        edge_width = 0.2
        layout_k = 0.3
    else:  # Large graphs (600+)
        node_size = 2
        edge_width = 0.1
        layout_k = 0.5
    
    # Plot Gen1 and closest S1
    for i in range(3):
        # Gen1
        ax = axes[0, i*2]
        if i < len(gen1_graphs) and gen1_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(gen1_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(gen1_graphs[i], pos, ax=ax, node_size=node_size, node_color='cyan',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'Gen1 #{i+1} ({gen1_graphs[i].number_of_nodes()}n/{gen1_graphs[i].number_of_edges()}e)')
        ax.axis('off')
        
        # Closest S1
        ax = axes[0, i*2+1]
        if i < len(closest_s1) and closest_s1[i] is not None and closest_s1[i].number_of_nodes() > 0:
            pos = nx.spring_layout(closest_s1[i], seed=42, k=layout_k, iterations=50)
            nx.draw(closest_s1[i], pos, ax=ax, node_size=node_size, node_color='lightblue',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            sim = sims_s1[i] if i < len(sims_s1) else 0
            ax.set_title(f'S1 (WL={sim:.3f}, {closest_s1[i].number_of_nodes()}n/{closest_s1[i].number_of_edges()}e)')
        ax.axis('off')
    
    # Plot Gen2 and closest S2
    for i in range(3):
        # Gen2
        ax = axes[1, i*2]
        if i < len(gen2_graphs) and gen2_graphs[i].number_of_nodes() > 0:
            pos = nx.spring_layout(gen2_graphs[i], seed=42, k=layout_k, iterations=50)
            nx.draw(gen2_graphs[i], pos, ax=ax, node_size=node_size, node_color='salmon',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            ax.set_title(f'Gen2 #{i+1} ({gen2_graphs[i].number_of_nodes()}n/{gen2_graphs[i].number_of_edges()}e)')
        ax.axis('off')
        
        # Closest S2
        ax = axes[1, i*2+1]
        if i < len(closest_s2) and closest_s2[i] is not None and closest_s2[i].number_of_nodes() > 0:
            pos = nx.spring_layout(closest_s2[i], seed=42, k=layout_k, iterations=50)
            nx.draw(closest_s2[i], pos, ax=ax, node_size=node_size, node_color='lightcoral',
                   with_labels=False, edge_color='gray', width=edge_width, alpha=0.6)
            sim = sims_s2[i] if i < len(sims_s2) else 0
            ax.set_title(f'S2 (WL={sim:.3f}, {closest_s2[i].number_of_nodes()}n/{closest_s2[i].number_of_edges()}e)')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = output_dir / f'4source_n{n_nodes}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def create_aggregate_plots(all_results, output_dir):
    """Create aggregate plots across all node sizes."""
    import matplotlib.pyplot as plt
    
    if not all_results:
        return
    
    # Extract data
    n_values = [r['n_nodes'] for r in all_results]
    gen_wl_means = [np.mean(r['generalization_wl']) for r in all_results]
    gen_wl_stds = [np.std(r['generalization_wl']) for r in all_results]
    mem_wl_means = [np.mean(r['memorization_wl']) for r in all_results]
    mem_wl_stds = [np.std(r['memorization_wl']) for r in all_results]
    
    # Plot 1: WL Similarity vs Node Size
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(n_values, gen_wl_means, yerr=gen_wl_stds, marker='o', label='Generalization (Gen1↔Gen2)', capsize=5)
    ax.errorbar(n_values, mem_wl_means, yerr=mem_wl_stds, marker='s', label='Memorization (Gen↔Train)', capsize=5)
    ax.set_xlabel('Number of Nodes (n)', fontsize=25)
    ax.set_ylabel('WL Similarity', fontsize=25)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_wl_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: aggregate_wl_similarity.png")
    
    # Plot 2: Empty Graph Percentage
    if 'empty_gen1_pct' in all_results[0]:
        empty_pcts = [(r['empty_gen1_pct'] + r['empty_gen2_pct']) / 2 for r in all_results]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(n_values, empty_pcts, marker='o', color='red')
        ax.set_xlabel('Number of Nodes (n)', fontsize=25)
        ax.set_ylabel('Empty Graphs (%)', fontsize=25)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregate_empty_graphs.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: aggregate_empty_graphs.png")


def load_models(checkpoint_dir: Path, n_nodes: int, N: int = 2500):
    """Load saved autoencoder and denoiser models.
    
    Args:
        checkpoint_dir: Directory containing saved checkpoints
        n_nodes: Graph size
        N: Training set size (default 2500)
        
    Returns:
        autoencoder_gen1, denoise_gen1, autoencoder_gen2, denoise_gen2, betas
    """
    exp_dir = checkpoint_dir / f"n_{n_nodes}" / f"N_{N}"
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {exp_dir}")
    
    # Check for checkpoint files
    ae1_path = exp_dir / "autoencoder_Gen1_N2500.pth.tar"
    dn1_path = exp_dir / "denoise_Gen1_N2500.pth.tar"
    ae2_path = exp_dir / "autoencoder_Gen2_N2500.pth.tar"
    dn2_path = exp_dir / "denoise_Gen2_N2500.pth.tar"
    
    for path in [ae1_path, dn1_path, ae2_path, dn2_path]:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    print(f"\n{'='*80}")
    print(f"Loading models for n={n_nodes}")
    print(f"{'='*80}")
    print(f"  Autoencoder Gen1: {ae1_path.name}")
    print(f"  Denoiser Gen1:    {dn1_path.name}")
    print(f"  Autoencoder Gen2: {ae2_path.name}")
    print(f"  Denoiser Gen2:    {dn2_path.name}")
    
    # Load dataset to get feature dimension
    data_list, data_path = load_dataset(n_nodes)
    
    # Determine input feature dimension
    if data_list is not None and len(data_list) > 0:
        # Legacy format: list of PyG Data objects
        input_feat_dim = data_list[0].x.shape[1] if hasattr(data_list[0], 'x') and data_list[0].x is not None else 2
    else:
        # Seeded format: load one sample to check
        from pathlib import Path
        seeded_dir = Path(data_path)
        if seeded_dir.exists() and (seeded_dir / 'S1.pt').exists():
            s1_sample = torch.load(seeded_dir / 'S1.pt', map_location='cpu', weights_only=False)
            if len(s1_sample) > 0:
                data_pyg, _ = s1_sample[0]
                input_feat_dim = data_pyg.x.shape[1] if hasattr(data_pyg, 'x') and data_pyg.x is not None else 2
            else:
                input_feat_dim = 2  # Default fallback
        else:
            input_feat_dim = 2  # Default fallback
    
    print(f"  Input feature dim: {input_feat_dim}")
    
    # Load one checkpoint to detect training configuration (bias, n_max_nodes)
    print(f"\n  Detecting training configuration from checkpoint...")
    ae1_ckpt_test = torch.load(ae1_path, map_location='cpu', weights_only=False)
    
    # Detect USE_BIAS from checkpoint keys
    has_bias_in_checkpoint = any('bias' in k for k in ae1_ckpt_test['state_dict'].keys())
    
    # Detect N_MAX_NODES from decoder output size
    # decoder.mlp.2.weight (or last layer) has shape [n_edges_ut, hidden_dim]
    # where n_edges_ut = n_max_nodes * (n_max_nodes - 1) // 2
    decoder_last_layer_key = [k for k in ae1_ckpt_test['state_dict'].keys() if 'decoder.mlp' in k and 'weight' in k][-1]
    n_edges_ut = ae1_ckpt_test['state_dict'][decoder_last_layer_key].shape[0]
    
    # Solve: n_edges_ut = n * (n - 1) / 2
    # => n^2 - n - 2*n_edges_ut = 0
    # => n = (1 + sqrt(1 + 8*n_edges_ut)) / 2
    n_max_nodes_detected = int((1 + np.sqrt(1 + 8 * n_edges_ut)) / 2)
    
    # Detect hidden dimensions from autoencoder
    # decoder.mlp.0.weight has shape [hidden_dim_decoder, latent_dim]
    hidden_dim_decoder_detected = ae1_ckpt_test['state_dict']['decoder.mlp.0.weight'].shape[0]
    
    # Load denoiser checkpoint to detect its hidden dim
    dn1_ckpt_test = torch.load(dn1_path, map_location='cpu', weights_only=False)
    # time_mlp.1.weight has shape [hidden_dim_denoise, hidden_dim_denoise]
    hidden_dim_denoise_detected = dn1_ckpt_test['state_dict']['time_mlp.1.weight'].shape[0]
    
    # Try to detect TIMESTEPS from checkpoint metadata or use default
    timesteps_detected = dn1_ckpt_test.get('timesteps', None)
    if timesteps_detected is None:
        # Try to infer from checkpoint metadata
        if 'config' in dn1_ckpt_test and 'timesteps' in dn1_ckpt_test['config']:
            timesteps_detected = dn1_ckpt_test['config']['timesteps']
        else:
            # Default: most LatestWorking checkpoints used 500 timesteps
            timesteps_detected = 500
            print(f"  ⚠️  TIMESTEPS not found in checkpoint, using default: {timesteps_detected}")
    
    print(f"  Detected USE_BIAS: {has_bias_in_checkpoint}")
    print(f"  Detected N_MAX_NODES: {n_max_nodes_detected} (n_edges_ut={n_edges_ut})")
    print(f"  Detected HIDDEN_DIM_DECODER: {hidden_dim_decoder_detected}")
    print(f"  Detected HIDDEN_DIM_DENOISE: {hidden_dim_denoise_detected}")
    print(f"  Detected TIMESTEPS: {timesteps_detected}")
    
    # Use detected settings for this model
    use_bias_actual = has_bias_in_checkpoint
    n_max_nodes_actual = n_max_nodes_detected
    hidden_dim_decoder_actual = hidden_dim_decoder_detected
    hidden_dim_denoise_actual = hidden_dim_denoise_detected
    timesteps_actual = timesteps_detected
    
    print(f"  Creating models with detected configuration...")
    
    # Create models with detected settings
    autoencoder_gen1 = VariationalAutoEncoder(
        input_dim=input_feat_dim,
        hidden_dim_enc=HIDDEN_DIM_ENCODER,
        hidden_dim_dec=hidden_dim_decoder_actual,
        latent_dim=LATENT_DIM,
        n_layers_enc=2,
        n_layers_dec=3,
        n_max_nodes=n_max_nodes_actual,
        use_bias=use_bias_actual
    ).to(device)
    
    autoencoder_gen2 = VariationalAutoEncoder(
        input_dim=input_feat_dim,
        hidden_dim_enc=HIDDEN_DIM_ENCODER,
        hidden_dim_dec=hidden_dim_decoder_actual,
        latent_dim=LATENT_DIM,
        n_layers_enc=2,
        n_layers_dec=3,
        n_max_nodes=n_max_nodes_actual,
        use_bias=use_bias_actual
    ).to(device)
    
    denoise_gen1 = DenoiseNN(
        input_dim=LATENT_DIM,
        hidden_dim=hidden_dim_denoise_actual,
        n_layers=3,
        n_cond=N_PROPERTIES,
        d_cond=128,
        use_bias=use_bias_actual
    ).to(device)
    
    denoise_gen2 = DenoiseNN(
        input_dim=LATENT_DIM,
        hidden_dim=hidden_dim_denoise_actual,
        n_layers=3,
        n_cond=N_PROPERTIES,
        d_cond=128,
        use_bias=use_bias_actual
    ).to(device)
    
    # Load checkpoints (reuse ae1_ckpt_test already loaded, reuse dn1_ckpt_test)
    autoencoder_gen1.load_state_dict(ae1_ckpt_test['state_dict'])
    autoencoder_gen1.eval()
    
    denoise_gen1.load_state_dict(dn1_ckpt_test['state_dict'])
    denoise_gen1.eval()
    
    ae2_ckpt = torch.load(ae2_path, map_location=device, weights_only=False)
    autoencoder_gen2.load_state_dict(ae2_ckpt['state_dict'])
    autoencoder_gen2.eval()
    
    dn2_ckpt = torch.load(dn2_path, map_location=device, weights_only=False)
    denoise_gen2.load_state_dict(dn2_ckpt['state_dict'])
    denoise_gen2.eval()
    
    # Create beta schedule with detected timesteps
    betas = linear_beta_schedule(timesteps=timesteps_actual)
    
    print(f"✅ Models loaded successfully!")
    
    return autoencoder_gen1, denoise_gen1, autoencoder_gen2, denoise_gen2, betas


def evaluate_models(checkpoint_dir: Path, n_nodes: int, output_dir: Path, N: int = 2500, K_NEAREST: int = 100):
    """Run evaluation on saved models for a specific node size.
    
    Args:
        checkpoint_dir: Directory with saved checkpoints
        n_nodes: Graph size
        output_dir: Where to save results
        N: Training set size
        K_NEAREST: Number of nearest neighbors for memorization check
    """
    from tqdm import tqdm
    
    # Load models
    autoencoder_gen1, denoise_gen1, autoencoder_gen2, denoise_gen2, betas = load_models(
        checkpoint_dir, n_nodes, N
    )
    
    # Load dataset
    print(f"\n--- Loading Dataset ---")
    data_list, data_path = load_dataset(n_nodes)
    
    # Load or create splits
    from pathlib import Path as PathLib
    cache_path = PathLib(data_path).parent / "cache" / f"labelhomophily0.2_{n_nodes}nodes_test_stats_seed42_size100.pt" if not PathLib(data_path).is_dir() else None
    
    # Try to load precomputed splits
    from main_nodesize import _load_precomputed_splits, shuffle_and_split_dataset
    split_config = _load_precomputed_splits(n_nodes, data_path, 100, 42, cache_path)
    
    if split_config is None:
        print(f"  No precomputed splits found, creating new splits...")
        split_config = shuffle_and_split_dataset(
            data_list,
            test_size=100,
            seed=42,
            stats_cache_path=cache_path
        )
    
    # Normalize keys for backward compatibility (seeded format uses 'S1'/'S2', old format uses 'S1_pool'/'S2_pool')
    if 'S1' in split_config and 'S1_pool' not in split_config:
        split_config['S1_pool'] = split_config['S1']
        split_config['S2_pool'] = split_config['S2']
    
    # Add missing 'seed' key if not present (seeded format doesn't include it)
    if 'seed' not in split_config:
        split_config['seed'] = 42  # Default seed used by seeded generator
    
    S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices = create_splits(split_config, N)
    
    print(f"  S1: {len(S1)} graphs")
    print(f"  S2: {len(S2)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    # Create output directory
    exp_dir = output_dir / f"n_{n_nodes}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    print(f"\n--- Generating & Evaluating ---")
    
    # Metrics storage
    generalization_scores = []
    memorization_scores_gen1 = []
    memorization_scores_gen2 = []
    empty_gen1_count = 0
    empty_gen2_count = 0
    
    # Precompute degree histograms
    print(f"Preparing {len(S1)} training graphs for memorization check (on-the-fly conversion)")
    n_bins = n_nodes
    S1_degree_hists = _precompute_degree_hists_from_pyg(S1, n_bins)
    S2_degree_hists = _precompute_degree_hists_from_pyg(S2, n_bins)
    print(f"Precomputed degree histograms for {len(S1)} S1 + {len(S2)} S2 graphs")
    
    # Prepare to accumulate generated graphs
    all_gen1_graphs = []
    all_gen2_graphs = []
    
    for i, test_graph in enumerate(tqdm(test_graphs, desc=f"Testing (n={n_nodes}, N={N})")):
        # Get conditioning stats
        if test_stats_cache is not None:
            cond_stats = test_stats_cache[i:i+1]
        else:
            cond_stats = test_graph.stats[:, :N_PROPERTIES]
        
        # Generate samples
        G1_samples, _, _ = generate_graphs(autoencoder_gen1, denoise_gen1, cond_stats, betas, num_samples=5)
        G2_samples, _, _ = generate_graphs(autoencoder_gen2, denoise_gen2, cond_stats, betas, num_samples=5)
        
        # Debug first test case
        if i == 0:
            print(f"\nFirst test case debug:")
            print(f"  Gen1 samples: {[f'{g.number_of_nodes()}n/{g.number_of_edges()}e' for g in G1_samples]}")
            print(f"  Gen2 samples: {[f'{g.number_of_nodes()}n/{g.number_of_edges()}e' for g in G2_samples]}")
        
        # Compute metrics for each sample
        for sample_idx in range(min(len(G1_samples), len(G2_samples))):
            g1 = G1_samples[sample_idx]
            g2 = G2_samples[sample_idx]
            all_gen1_graphs.append(g1)
            all_gen2_graphs.append(g2)
            
            # Track empty graphs
            if g1.number_of_nodes() == 0:
                empty_gen1_count += 1
            if g2.number_of_nodes() == 0:
                empty_gen2_count += 1
            
            # Generalization: Gen1 vs Gen2
            sim = wl_similarity_degree_only(g1, g2)
            generalization_scores.append(sim)
            
            # Debug first few
            if i == 0 and sample_idx < 2:
                print(f"  Gen1 vs Gen2 sample {sample_idx}: WL(degree-only) = {sim:.4f}")
        
        # Memorization: Gen1 vs S1, Gen2 vs S2
        
        for g in G1_samples:
            _, sim_g1 = find_closest_graph_in_training_degree_only_k(g, S1, S1_degree_hists, n_bins, K_NEAREST)
            memorization_scores_gen1.append(sim_g1)
        
        for g in G2_samples:
            _, sim_g2 = find_closest_graph_in_training_degree_only_k(g, S2, S2_degree_hists, n_bins, K_NEAREST)
            memorization_scores_gen2.append(sim_g2)
        
        # Debug for first test
        if i < 2:
            gen_has_feat = False
            if G1_samples[0].number_of_nodes() > 0:
                first_node = list(G1_samples[0].nodes())[0]
                gen_has_feat = 'feature_vector' in G1_samples[0].nodes[first_node]
            
            dbg_closest, dbg_sim = find_closest_graph_in_training_degree_only_k(G1_samples[0], S1, S1_degree_hists, n_bins, min(10, K_NEAREST))
            if dbg_closest is not None:
                print(f"  Test {i}: Gen1 sample0 ({G1_samples[0].number_of_nodes()}n, {G1_samples[0].number_of_edges()}e, has_feat={gen_has_feat}) "
                      f"vs closest S1 WLdeg={dbg_sim:.4f}")
            
            if i == 0 and len(S1) > 0:
                test_sim = wl_similarity_degree_only(G1_samples[0], _pyg_to_nx(S1[0]))
                print(f"  Direct WL test (degree-only) Gen1 vs first S1: {test_sim:.4f}")
    
    # Compute statistics
    memorization_scores = memorization_scores_gen1 + memorization_scores_gen2
    
    total_gen1_graphs = len(all_gen1_graphs)
    total_gen2_graphs = len(all_gen2_graphs)
    empty_gen1_pct = 100 * empty_gen1_count / total_gen1_graphs if total_gen1_graphs > 0 else 0
    empty_gen2_pct = 100 * empty_gen2_count / total_gen2_graphs if total_gen2_graphs > 0 else 0
    
    # Print results
    print(f"\nResults:")
    print(f"  Generalization (Gen1 vs Gen2) [WL deg-only]: {np.mean(generalization_scores):.4f} ± {np.std(generalization_scores):.4f}")
    print(f"  Memorization (symmetric) [WL deg-only]:      {np.mean(memorization_scores):.4f} ± {np.std(memorization_scores):.4f}")
    print(f"\n  [WARNING] Empty Graph Statistics:")
    print(f"    Gen1 empty: {empty_gen1_count}/{total_gen1_graphs} ({empty_gen1_pct:.1f}%)")
    print(f"    Gen2 empty: {empty_gen2_count}/{total_gen2_graphs} ({empty_gen2_pct:.1f}%)")
    
    # Generate example visualizations
    print(f"\n--- Generating example graphs for visualization ---")
    torch.manual_seed(999)
    example_conditioning = test_stats_cache[0:1] if isinstance(test_stats_cache, torch.Tensor) else test_graphs[0].stats[:, :N_PROPERTIES]
    
    example_gen1, _, _ = generate_graphs(autoencoder_gen1, denoise_gen1, example_conditioning, betas, num_samples=3)
    torch.manual_seed(999)
    example_gen2, _, _ = generate_graphs(autoencoder_gen2, denoise_gen2, example_conditioning, betas, num_samples=3)
    
    # Find closest training matches
    closest_s1_for_gen1 = []
    closest_s1_sims = []
    for G in example_gen1:
        G_closest, sim = find_closest_graph_in_training_degree_only_k(G, S1, S1_degree_hists, n_bins, K_NEAREST)
        closest_s1_for_gen1.append(G_closest)
        closest_s1_sims.append(sim)
    
    closest_s2_for_gen2 = []
    closest_s2_sims = []
    for G in example_gen2:
        G_closest, sim = find_closest_graph_in_training_degree_only_k(G, S2, S2_degree_hists, n_bins, K_NEAREST)
        closest_s2_for_gen2.append(G_closest)
        closest_s2_sims.append(sim)
    
    # Example training graphs
    if len(S1) > 0:
        idxs_s1 = [0, min(10, len(S1)-1), min(20, len(S1)-1)]
        example_s1 = [_pyg_to_nx(S1[i]) for i in idxs_s1]
    else:
        example_s1 = []
    
    if len(S2) > 0:
        idxs_s2 = [0, min(10, len(S2)-1), min(20, len(S2)-1)]
        example_s2_graphs = [_pyg_to_nx(S2[i]) for i in idxs_s2]
    else:
        example_s2_graphs = []
    
    # Create visualizations
    print(f"\n--- Creating visualizations for n={n_nodes} ---")
    print(f"  Creating GT vs Generated for n={n_nodes}...")
    visualize_gt_vs_generated(example_s1, example_s2_graphs, example_gen1, example_gen2, n_nodes, fig_dir)
    
    print(f"  Creating 4-source example graphs for n={n_nodes}...")
    visualize_gen_vs_closest_train(
        example_gen1, example_gen2,
        closest_s1_for_gen1, closest_s2_for_gen2,
        closest_s1_sims, closest_s2_sims,
        n_nodes, fig_dir
    )
    
    # Return metrics for aggregate plotting
    return {
        'n_nodes': n_nodes,
        'N': N,
        'complexity_ratio': n_nodes / N,
        'generalization_wl': generalization_scores,
        'memorization_wl': memorization_scores,
        'empty_gen1_pct': empty_gen1_pct,
        'empty_gen2_pct': empty_gen2_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models without retraining")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing saved model checkpoints (e.g., outputs/nodesize_study/LatestWorking)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save evaluation results (default: checkpoint-dir with _eval suffix)")
    parser.add_argument("--node-sizes", type=int, nargs="+", default=[5, 10, 20, 30, 50, 100],
                        help="Node sizes to evaluate (default: 5 10 20 30 50 100)")
    parser.add_argument("--k-nearest", type=int, default=100,
                        help="Number of nearest neighbors for memorization check (default: 100)")
    parser.add_argument("--no-bias", action="store_true",
                        help="Disable bias in models (must match training)")
    
    args = parser.parse_args()
    
    # Update global USE_BIAS
    if args.no_bias:
        import main_nodesize
        main_nodesize.USE_BIAS = False
        print("⚠️  Running with bias=False for all models")
    
    checkpoint_dir = Path(args.checkpoint_dir).resolve()  # Convert to absolute path
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print(f"   (Resolved from: {args.checkpoint_dir})")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Evaluating Saved Models (No Retraining)")
    print(f"{'='*80}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Node sizes: {args.node_sizes}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Evaluate each node size
    all_results = []
    for n_nodes in args.node_sizes:
        try:
            result = evaluate_models(checkpoint_dir, n_nodes, output_dir, K_NEAREST=args.k_nearest)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping n={n_nodes}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error evaluating n={n_nodes}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\n❌ No results generated - check checkpoint paths!")
        sys.exit(1)
    
    # Convert list to dict keyed by n_nodes (matching main_nodesize.py format)
    all_results_dict = {r['n_nodes']: r for r in all_results}
    
    # Save raw results for later plotting
    import pickle
    pkl_path = output_dir / "all_results.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results_dict, f)
    print(f"✅ Saved raw results to: {pkl_path}")
    
    # Create aggregate plots
    print(f"\n--- Creating aggregate visualizations ---")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    create_aggregate_plots(all_results, figures_dir)
    
    # Save summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint directory: {checkpoint_dir}\n")
        f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Node sizes evaluated: {[r['n_nodes'] for r in all_results]}\n\n")
        
        for result in all_results:
            f.write(f"\nn={result['n_nodes']}, N={result['N']}, ratio={result['complexity_ratio']:.4f}\n")
            f.write(f"  Generalization (WL deg-only): {np.mean(result['generalization_wl']):.4f} ± {np.std(result['generalization_wl']):.4f}\n")
            f.write(f"  Memorization (WL deg-only):   {np.mean(result['memorization_wl']):.4f} ± {np.std(result['memorization_wl']):.4f}\n")
            f.write(f"  Empty graphs: Gen1={result['empty_gen1_pct']:.1f}%, Gen2={result['empty_gen2_pct']:.1f}%\n")
    
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {figures_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
