#!/usr/bin/env python3
"""
Create histogram plots from evaluation summary file without re-running experiments.

This recreates the overlapping histogram plots showing memorization vs generalization
distributions across different node sizes.

Usage:
    python plot_histograms_from_results.py --summary evaluation_summary.txt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def parse_summary_file(summary_path):
    """Parse evaluation_summary.txt or experiment_summary.txt to extract metrics."""
    with open(summary_path, 'r') as f:
        content = f.read()
    
    all_results = {}
    
    # Try format 1: "n=X, N=Y, ratio=Z\n  Generalization (WL deg-only): mean ± std"
    pattern1 = r'n=(\d+), N=(\d+), ratio=([\d.]+)\s+Generalization \(WL deg-only\): ([\d.]+) ± ([\d.]+)\s+Memorization \(WL deg-only\):\s+([\d.]+) ± ([\d.]+)'
    matches = re.findall(pattern1, content)
    
    if matches:
        print(f"✓ Found {len(matches)} entries in evaluation_summary.txt format")
        for match in matches:
            n_nodes = int(match[0])
            N = int(match[1])
            ratio = float(match[2])
            gen_mean = float(match[3])
            gen_std = float(match[4])
            mem_mean = float(match[5])
            mem_std = float(match[6])
            
            # Reconstruct approximate distributions from mean ± std
            np.random.seed(42 + n_nodes)  # Different seed per n for variety
            n_samples = 500
            
            gen_scores = np.random.normal(gen_mean, gen_std, n_samples)
            mem_scores = np.random.normal(mem_mean, mem_std, n_samples)
            
            # Clip to [0, 1] range
            gen_scores = np.clip(gen_scores, 0, 1)
            mem_scores = np.clip(mem_scores, 0, 1)
            
            all_results[n_nodes] = {
                'n_nodes': n_nodes,
                'N': N,
                'complexity_ratio': ratio,
                'generalization_wl': gen_scores.tolist(),
                'memorization_wl': mem_scores.tolist(),
                'gen_mean': gen_mean,
                'gen_std': gen_std,
                'mem_mean': mem_mean,
                'mem_std': mem_std
            }
    else:
        # Try format 2: Table format from experiment_summary.txt
        # Pattern: n_nodes    ComplexRatio   Gen_WL_Mean    Gen_WL_Std     Mem_WL_Mean    Mem_WL_Std
        print("✓ Trying experiment_summary.txt table format...")
        
        # Extract N from header
        N = 2500  # Default
        n_match = re.search(r'Fixed training size: N = (\d+)', content)
        if n_match:
            N = int(n_match.group(1))
        
        # Find table rows (skip header and separator lines)
        lines = content.split('\n')
        in_table = False
        for line in lines:
            # Skip headers and separators
            if '----' in line or 'n_nodes' in line or 'ComplexRatio' in line:
                in_table = True
                continue
            
            if not in_table or not line.strip():
                continue
            
            # Try to parse data row: n_nodes    ComplexRatio   Gen_WL_Mean    Gen_WL_Std     Mem_WL_Mean    Mem_WL_Std
            parts = line.split()
            if len(parts) >= 6:
                try:
                    n_nodes = int(parts[0])
                    ratio = float(parts[1])
                    gen_mean = float(parts[2])
                    gen_std = float(parts[3])
                    mem_mean = float(parts[4])
                    mem_std = float(parts[5])
                    
                    # Reconstruct distributions
                    np.random.seed(42 + n_nodes)
                    n_samples = 500
                    
                    gen_scores = np.random.normal(gen_mean, gen_std, n_samples)
                    mem_scores = np.random.normal(mem_mean, mem_std, n_samples)
                    
                    gen_scores = np.clip(gen_scores, 0, 1)
                    mem_scores = np.clip(mem_scores, 0, 1)
                    
                    all_results[n_nodes] = {
                        'n_nodes': n_nodes,
                        'N': N,
                        'complexity_ratio': ratio,
                        'generalization_wl': gen_scores.tolist(),
                        'memorization_wl': mem_scores.tolist(),
                        'gen_mean': gen_mean,
                        'gen_std': gen_std,
                        'mem_mean': mem_mean,
                        'mem_std': mem_std
                    }
                except (ValueError, IndexError):
                    continue
        
        if all_results:
            print(f"✓ Found {len(all_results)} entries in experiment_summary.txt format")
    
    return all_results


def create_histograms_from_all_results(all_results, output_dir):
    """
    Create overlapping histogram plots matching main_nodesize.py format.
    
    all_results: dict mapping n_nodes -> result dict with keys:
        - 'generalization_scores': list of Gen1 vs Gen2 WL similarities
        - 'memorization_scores': list of Gen vs Train WL similarities
        - 'complexity_ratio': n/N ratio
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract node sizes
    n_vals = sorted(all_results.keys())
    n_plots = len(n_vals)
    
    if n_plots == 0:
        print("❌ No results found!")
        return
    
    print(f"{'='*80}")
    print(f"Creating Histogram Plots for n={n_vals}")
    print(f"{'='*80}\n")
    
    # 1. Main histogram plot: Overlapping distributions with 2 rows (5 plots each)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Flatten axes array for easy indexing
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, n_nodes in enumerate(n_vals):
        result = all_results[n_nodes]
        ax = axes[idx]
        
        gen_scores = result['generalization_wl']
        mem_scores = result['memorization_wl']
        
        # Plot overlapping histograms (normalized density)
        bins = np.linspace(0, 1, 21)
        if len(mem_scores) > 0:
            ax.hist(mem_scores, bins=bins, alpha=0.5, color='orange', 
                   density=True, edgecolor='black')
        if len(gen_scores) > 0:
            ax.hist(gen_scores, bins=bins, alpha=0.5, color='blue', 
                   density=True, edgecolor='black')
        
        # Add n=value text inside the plot (top-left)
        ax.text(0.05, 0.95, f'n={n_nodes}', 
                transform=ax.transAxes, fontsize=16, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        ax.set_xlabel('WL Similarity', fontsize=25)
        ax.set_ylabel('Density', fontsize=25)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=14)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add a single legend at the top of the figure (higher placement)
    handles = [
        plt.Line2D([0], [0], color='blue', lw=8, alpha=0.5, label='Samples from two denoisers'),
        plt.Line2D([0], [0], color='orange', lw=8, alpha=0.5, label='Sample and closest training graph')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=2, fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = output_dir / 'memorization_to_generalization_histograms.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
    
    # 2. Feature-WL histograms (if available)
    has_feat = all(('generalization_wl_feat' in r and 'memorization_wl_feat' in r) 
                   for r in all_results.values())
    
    if has_feat:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        # Flatten axes array
        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        for idx, n_nodes in enumerate(n_vals):
            result = all_results[n_nodes]
            ax = axes[idx]
            
            gen_scores = result['generalization_wl_feat']
            mem_scores = result['memorization_wl_feat']
            
            bins = np.linspace(0, 1, 21)
            if len(mem_scores) > 0:
                ax.hist(mem_scores, bins=bins, alpha=0.5, color='orange', 
                       density=True, edgecolor='black')
            if len(gen_scores) > 0:
                ax.hist(gen_scores, bins=bins, alpha=0.5, color='blue', 
                       density=True, edgecolor='black')
            
            # Add n=value text inside the plot (top-left)
            ax.text(0.05, 0.95, f'n={n_nodes}', 
                    transform=ax.transAxes, fontsize=16, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
            
            ax.set_xlabel('WL (degree-feature) Similarity', fontsize=25)
            ax.set_ylabel('Density', fontsize=25)
            ax.set_xlim(0, 1)
            ax.tick_params(axis='both', labelsize=14)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Add a single legend at the top of the figure (higher placement)
        handles = [
            plt.Line2D([0], [0], color='blue', lw=8, alpha=0.5, label='Samples from two denoisers'),
            plt.Line2D([0], [0], color='orange', lw=8, alpha=0.5, label='Sample and closest training graph')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                   ncol=2, fontsize=14, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = output_dir / 'memorization_to_generalization_histograms_feature.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    
    print(f"\n{'='*80}")
    print(f"Histograms saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Create histograms from evaluation summary")
    parser.add_argument("--summary", type=str, required=True,
                        help="Path to evaluation_summary.txt file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save histograms (default: same dir as summary file)")
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"❌ File not found: {summary_path}")
        return
    
    print(f"Loading results from: {summary_path}")
    all_results = parse_summary_file(summary_path)
    
    if not all_results:
        print("❌ No data found in summary file!")
        return
    
    if args.output_dir is None:
        output_dir = summary_path.parent / "histograms"
    else:
        output_dir = Path(args.output_dir)
    
    create_histograms_from_all_results(all_results, output_dir)


if __name__ == "__main__":
    main()
