"""
Run Jacobian analysis on trained graph diffusion models

Usage:
    python run_jacobian_analysis.py --checkpoint checkpoints/DF1_n20_best.pt --node_size 20
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from lightweight_graphmaker.sample import GraphSampler
from lightweight_graphmaker.jacobian_analysis import (
    calc_jacobian_graph, calc_jacobian_row_graph,
    pca_denoised_graphs, analyze_jacobian_properties,
    compute_effective_rank
)


def plot_jacobian_heatmap(jacob, save_path=None):
    """Visualize Jacobian matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(jacob.cpu().numpy(), cmap='RdBu_r', aspect='auto',
                   vmin=-jacob.abs().max(), vmax=jacob.abs().max())
    
    ax.set_xlabel('Input Edge Index', fontsize=14)
    ax.set_ylabel('Output Edge Index', fontsize=14)
    ax.set_title('Jacobian Matrix of Graph Denoiser', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('∂output[i] / ∂input[j]', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Jacobian heatmap to {save_path}")
    
    return fig


def plot_singular_values(singular_values, save_path=None):
    """Plot singular value spectrum"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(singular_values.cpu().numpy(), linewidth=2, color='steelblue')
    axes[0].set_xlabel('Index', fontsize=14)
    axes[0].set_ylabel('Singular Value', fontsize=14)
    axes[0].set_title('Singular Value Spectrum', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].semilogy(singular_values.cpu().numpy(), linewidth=2, color='darkred')
    axes[1].set_xlabel('Index', fontsize=14)
    axes[1].set_ylabel('Singular Value (log scale)', fontsize=14)
    axes[1].set_title('Singular Value Spectrum (log)', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved singular value plot to {save_path}")
    
    return fig


def plot_eigenvalue_decay(eigenvalues_dict, save_path=None):
    """Plot eigenvalue decay for different noise levels"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(eigenvalues_dict)))
    
    for (t, eigenvalues), color in zip(sorted(eigenvalues_dict.items()), colors):
        # Normalize
        normalized = eigenvalues / eigenvalues.sum()
        cumsum = torch.cumsum(normalized, dim=0)
        
        # Linear
        axes[0].plot(normalized.cpu().numpy()[:50], label=f't={t}', 
                    linewidth=2, color=color)
        
        # Cumulative
        axes[1].plot(cumsum.cpu().numpy()[:50], label=f't={t}',
                    linewidth=2, color=color)
    
    axes[0].set_xlabel('Component Index', fontsize=14)
    axes[0].set_ylabel('Normalized Eigenvalue', fontsize=14)
    axes[0].set_title('Eigenvalue Decay', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Component Index', fontsize=14)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=14)
    axes[1].set_title('Cumulative Variance', fontsize=16, fontweight='bold')
    axes[1].axhline(0.99, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved eigenvalue decay plot to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Jacobian analysis for graph denoiser')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--node_size', type=int, required=True,
                       help='Number of nodes (20, 50, 100, or 500)')
    parser.add_argument('--split', type=str, default='S1',
                       help='Which split to use (S1 or S2)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on')
    parser.add_argument('--noise_levels', type=int, nargs='+', 
                       default=[20, 40, 60, 80, 100],
                       help='Noise levels (timesteps) for PCA analysis')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples per noise level for PCA')
    parser.add_argument('--output_dir', type=str, default='jacobian_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("JACOBIAN ANALYSIS FOR GRAPH DENOISER")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Node size: {args.node_size}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    sampler = GraphSampler(args.checkpoint, args.device)
    model = sampler.model
    diffusion = sampler.diffusion
    
    # Load a clean graph from the dataset
    data_path = f'data/node_{args.node_size}/{args.split}.pt'
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    # Use first graph
    clean_graph, _ = data[0]
    clean_edge_index = clean_graph.edge_index
    Y = clean_graph.y.to(args.device)
    
    # Convert to edge vector
    clean_edge_vector = diffusion.adjacency_to_edge_vector(clean_edge_index).to(args.device)
    
    print(f"Clean graph: {clean_graph.num_nodes} nodes, {clean_graph.num_edges} edges")
    print(f"Edge vector dimension: {clean_edge_vector.shape[0]}")
    
    # ========== Analysis 1: Compute full Jacobian at different timesteps ==========
    print("\n" + "="*80)
    print("ANALYSIS 1: Jacobian Matrix at Different Noise Levels")
    print("="*80)
    
    jacobian_results = {}
    
    for t in [20, 50, 80]:
        print(f"\n--- Computing Jacobian at t={t} ---")
        
        # Add noise
        E_t, _, _ = diffusion.forward_diffusion(clean_edge_vector.clone(), t=t)
        E_t = E_t.to(args.device).float()  # Convert to float for edge flipping perturbations
        
        t_normalized = torch.tensor([t / diffusion.T], device=args.device)
        
        # Compute Jacobian
        jacob = calc_jacobian_graph(E_t, Y, t_normalized, model, diffusion)
        
        # Analyze
        analysis = analyze_jacobian_properties(jacob)
        jacobian_results[t] = {
            'jacobian': jacob,
            'analysis': analysis
        }
        
        # Plot Jacobian heatmap
        plot_jacobian_heatmap(jacob, 
            save_path=os.path.join(args.output_dir, f'jacobian_heatmap_t{t}.png'))
        plt.close()
        
        # Plot singular values
        plot_singular_values(analysis['singular_values'],
            save_path=os.path.join(args.output_dir, f'singular_values_t{t}.png'))
        plt.close()
    
    # ========== Analysis 2: PCA of denoised outputs ==========
    print("\n" + "="*80)
    print("ANALYSIS 2: PCA of Denoised Graphs")
    print("="*80)
    
    eigenvalues, eigenvectors, denoised_graphs = pca_denoised_graphs(
        clean_edge_vector=clean_edge_vector.clone(),
        Y=Y,
        model=model,
        diffusion=diffusion,
        noise_levels=args.noise_levels,
        num_samples=args.num_samples,
        device=args.device
    )
    
    # Plot eigenvalue decay
    plot_eigenvalue_decay(eigenvalues,
        save_path=os.path.join(args.output_dir, 'eigenvalue_decay.png'))
    plt.close()
    
    # Compute effective ranks
    print("\n" + "="*80)
    print("EFFECTIVE RANK ANALYSIS")
    print("="*80)
    
    for t in sorted(eigenvalues.keys()):
        for threshold in [0.90, 0.95, 0.99]:
            eff_rank, cumsum = compute_effective_rank(eigenvalues[t], threshold)
            print(f"t={t:3d}, threshold={threshold:.2f}: "
                  f"effective_rank={eff_rank:4d} / {len(eigenvalues[t])}")
    
    # ========== Save results ==========
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results = {
        'checkpoint': args.checkpoint,
        'node_size': args.node_size,
        'split': args.split,
        'jacobian_analysis': {
            t: {
                'condition_number': res['analysis']['condition_number'],
                'rank': res['analysis']['rank'],
                'effective_rank_99': res['analysis']['effective_rank_99'],
                'frobenius_norm': res['analysis']['frobenius_norm'],
                'off_diagonal_ratio': res['analysis']['off_diagonal_ratio']
            }
            for t, res in jacobian_results.items()
        },
        'pca_analysis': {
            'noise_levels': args.noise_levels,
            'num_samples': args.num_samples,
            'effective_ranks': {
                t: {
                    f'threshold_{int(th*100)}': compute_effective_rank(eigenvalues[t], th)[0]
                    for th in [0.90, 0.95, 0.99]
                }
                for t in eigenvalues.keys()
            }
        }
    }
    
    import json
    results_path = os.path.join(args.output_dir, 'jacobian_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
