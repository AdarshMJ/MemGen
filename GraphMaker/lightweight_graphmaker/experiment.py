"""
Complete experiment pipeline for memorization vs generalization study

Trains DF1 on S1, DF2 on S2 for different node sizes,
generates graphs, and evaluates WL similarity
"""
import torch
import os
import json
import argparse
from datetime import datetime

from train import train_model
from sample import generate_from_checkpoint
from dataset import GraphDataset
from wl_kernel import evaluate_memorization_vs_generalization


def run_experiment(node_size, 
                  hidden_dim=128,
                  num_layers=3,
                  num_timesteps=100,
                  batch_size=16,
                  lr=1e-3,
                  num_epochs=100,
                  num_generated=100,
                  device='cuda',
                  output_dir='experiments'):
    """
    Run complete experiment for a given node size
    
    Args:
        node_size: Number of nodes (20, 50, 100, 500)
        ... other hyperparameters ...
    
    Returns:
        results: Dict with WL similarity scores
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: node_size={node_size}")
    print("="*80)
    
    # Create experiment directory
    exp_name = f"n{node_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save experiment config
    config = {
        'node_size': node_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_timesteps': num_timesteps,
        'batch_size': batch_size,
        'lr': lr,
        'num_epochs': num_epochs,
        'num_generated': num_generated,
        'device': device
    }
    
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # ========== STEP 1: Train DF1 on S1 ==========
    print("\n" + "="*80)
    print("STEP 1: Training DF1 on S1 (high homophily)")
    print("="*80)
    
    s1_path = f'data/node_{node_size}/S1.pt'
    trainer_df1 = train_model(
        data_path=s1_path,
        model_name='DF1',
        node_size=node_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        device=device,
        save_dir=checkpoint_dir
    )
    
    # ========== STEP 2: Train DF2 on S2 ==========
    print("\n" + "="*80)
    print("STEP 2: Training DF2 on S2 (low homophily)")
    print("="*80)
    
    s2_path = f'data/node_{node_size}/S2.pt'
    trainer_df2 = train_model(
        data_path=s2_path,
        model_name='DF2',
        node_size=node_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        device=device,
        save_dir=checkpoint_dir
    )
    
    # ========== STEP 3: Generate from DF1 ==========
    print("\n" + "="*80)
    print(f"STEP 3: Generating {num_generated} graphs from DF1")
    print("="*80)
    
    df1_checkpoint = os.path.join(checkpoint_dir, f'DF1_n{node_size}_best.pt')
    gen1_path = os.path.join(exp_dir, f'gen1_n{node_size}.pt')
    
    gen1_graphs = generate_from_checkpoint(
        checkpoint_path=df1_checkpoint,
        num_samples=num_generated,
        output_path=gen1_path,
        device=device
    )
    
    # ========== STEP 4: Generate from DF2 ==========
    print("\n" + "="*80)
    print(f"STEP 4: Generating {num_generated} graphs from DF2")
    print("="*80)
    
    df2_checkpoint = os.path.join(checkpoint_dir, f'DF2_n{node_size}_best.pt')
    gen2_path = os.path.join(exp_dir, f'gen2_n{node_size}.pt')
    
    gen2_graphs = generate_from_checkpoint(
        checkpoint_path=df2_checkpoint,
        num_samples=num_generated,
        output_path=gen2_path,
        device=device
    )
    
    # ========== STEP 5: Load training data ==========
    print("\n" + "="*80)
    print("STEP 5: Loading training data")
    print("="*80)
    
    s1_dataset = GraphDataset(s1_path)
    s2_dataset = GraphDataset(s2_path)
    
    # Take subset for evaluation (to match num_generated)
    s1_graphs = [s1_dataset[i][0] for i in range(min(num_generated, len(s1_dataset)))]
    s2_graphs = [s2_dataset[i][0] for i in range(min(num_generated, len(s2_dataset)))]
    
    print(f"Loaded {len(s1_graphs)} graphs from S1")
    print(f"Loaded {len(s2_graphs)} graphs from S2")
    
    # ========== STEP 6: Evaluate WL similarity ==========
    print("\n" + "="*80)
    print("STEP 6: Evaluating Weisfeiler-Lehman similarity")
    print("="*80)
    
    # Create visualizations directory
    vis_dir = os.path.join(exp_dir, 'visualizations')
    
    results = evaluate_memorization_vs_generalization(
        gen1_graphs=gen1_graphs,
        gen2_graphs=gen2_graphs,
        s1_graphs=s1_graphs,
        s2_graphs=s2_graphs,
        n_iter=5,
        output_dir=vis_dir  # Pass output directory for visualizations
    )
    
    # ========== STEP 7: Compute Graph Statistics MSE ==========
    print("\n" + "="*80)
    print("STEP 7: Computing Graph Statistics MSE with Test Set")
    print("="*80)
    
    from graph_stats import compute_mse_with_test
    
    # Compute MSE for Gen1 vs Test
    print("\nGen1 (DF1) vs Test:")
    test_path = f'data/node_{node_size}/test.pt'
    mse_results_gen1, _, _ = compute_mse_with_test(gen1_graphs, test_path, verbose=True)
    
    # Compute MSE for Gen2 vs Test
    print("\nGen2 (DF2) vs Test:")
    mse_results_gen2, _, _ = compute_mse_with_test(gen2_graphs, test_path, verbose=True)
    
    # Add to results
    results['Gen1_stats_mse'] = mse_results_gen1
    results['Gen2_stats_mse'] = mse_results_gen2
    
    # Add training losses to results
    results['DF1_final_loss'] = trainer_df1.train_losses[-1]
    results['DF2_final_loss'] = trainer_df2.train_losses[-1]
    results['DF1_best_loss'] = trainer_df1.best_loss
    results['DF2_best_loss'] = trainer_df2.best_loss
    
    # Save results
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETE: node_size={node_size}")
    print("="*80)
    print(f"Experiment directory: {exp_dir}")
    print(f"\nTraining Losses:")
    print(f"  DF1 best loss: {results['DF1_best_loss']:.4f}")
    print(f"  DF2 best loss: {results['DF2_best_loss']:.4f}")
    print(f"\nWL Similarity:")
    print(f"  WLSim(Gen1, S1): {results['WLSim_Gen1_S1']:.4f}")
    print(f"  WLSim(Gen2, S2): {results['WLSim_Gen2_S2']:.4f}")
    print(f"  WLSim(Gen1, Gen2): {results['WLSim_Gen1_Gen2']:.4f}")
    print(f"  Gen/Mem Ratio: {results['gen_vs_mem_ratio']:.4f}")
    print(f"\nGraph Statistics MSE vs Test:")
    print(f"  Gen1 overall MSE: {results['Gen1_stats_mse']['overall_mse']:.6f}")
    print(f"  Gen2 overall MSE: {results['Gen2_stats_mse']['overall_mse']:.6f}")
    print("="*80 + "\n")
    
    return results, exp_dir


def run_full_study(node_sizes=[20, 50, 100, 500],
                   num_epochs_map={20: 100, 50: 150, 100: 200, 500: 300},
                   **kwargs):
    """
    Run experiments for all node sizes
    
    Args:
        node_sizes: List of node sizes to experiment on
        num_epochs_map: Dict mapping node_size to num_epochs
        **kwargs: Additional hyperparameters
    """
    print("\n" + "="*80)
    print("FULL MEMORIZATION VS GENERALIZATION STUDY")
    print("="*80)
    print(f"Node sizes: {node_sizes}")
    print(f"Epochs: {num_epochs_map}")
    print("="*80 + "\n")
    
    all_results = {}
    all_exp_dirs = {}
    
    for node_size in node_sizes:
        num_epochs = num_epochs_map.get(node_size, 100)
        
        try:
            results, exp_dir = run_experiment(
                node_size=node_size,
                num_epochs=num_epochs,
                **kwargs
            )
            
            all_results[f'n{node_size}'] = results
            all_exp_dirs[f'n{node_size}'] = exp_dir
            
        except Exception as e:
            print(f"\nERROR in experiment for node_size={node_size}:")
            print(f"{e}")
            import traceback
            traceback.print_exc()
    
    # Save summary of all experiments
    summary_dir = 'experiments/summary'
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, 
                                f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    summary = {
        'node_sizes': node_sizes,
        'results': all_results,
        'exp_dirs': all_exp_dirs
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("FULL STUDY COMPLETE")
    print("="*80)
    print(f"Summary saved to {summary_path}")
    print("\nResults across node sizes:")
    print("-"*80)
    print(f"{'Node Size':<12} {'Gen/Mem Ratio':<15} {'Status':<20}")
    print("-"*80)
    
    for node_size in node_sizes:
        key = f'n{node_size}'
        if key in all_results:
            ratio = all_results[key]['gen_vs_mem_ratio']
            if ratio > 1.2:
                status = "GENERALIZE"
            elif ratio < 0.8:
                status = "MEMORIZE"
            else:
                status = "INTERMEDIATE"
            print(f"{node_size:<12} {ratio:<15.4f} {status:<20}")
        else:
            print(f"{node_size:<12} {'ERROR':<15} {'FAILED':<20}")
    
    print("="*80 + "\n")
    
    return all_results, all_exp_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run memorization vs generalization experiments')
    
    # Experiment options
    parser.add_argument('--node_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help='Node sizes to experiment on')
    parser.add_argument('--single_node_size', type=int, default=None,
                        help='Run single experiment for this node size')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=100)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    
    # Generation
    parser.add_argument('--num_generated', type=int, default=100)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='experiments')
    
    args = parser.parse_args()
    
    if args.single_node_size is not None:
        # Run single experiment
        results, exp_dir = run_experiment(
            node_size=args.single_node_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_timesteps=args.num_timesteps,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            num_generated=args.num_generated,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        # Run full study
        all_results, all_exp_dirs = run_full_study(
            node_sizes=args.node_sizes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_timesteps=args.num_timesteps,
            batch_size=args.batch_size,
            lr=args.lr,
            num_generated=args.num_generated,
            device=args.device,
            output_dir=args.output_dir
        )
