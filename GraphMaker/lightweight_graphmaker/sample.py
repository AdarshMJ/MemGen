"""
Sampling and generation from trained diffusion models
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import os

from .diffusion import GraphDiffusion
from .model import BiasFreeDenoisingGNN


class GraphSampler:
    """
    Sample graphs from trained diffusion model
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to use
    """
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = checkpoint['config']
        self.config = config
        
        # Initialize diffusion
        E_marginal = torch.tensor(config['E_marginal'])
        Y_marginal = torch.tensor(config['Y_marginal'])
        
        self.diffusion = GraphDiffusion(
            T=config['num_timesteps'],
            E_marginal=E_marginal,
            Y_marginal=Y_marginal,
            num_nodes=config['num_nodes']
        ).to(self.device)
        
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        
        # Get actual num_classes from checkpoint state dict if available
        actual_num_classes = config['num_classes']
        if 'model_state_dict' in checkpoint:
            # Check the embedding layer size to get actual number of classes
            embedding_weight = checkpoint['model_state_dict'].get('label_embedding.weight')
            if embedding_weight is not None:
                actual_num_classes = embedding_weight.shape[0]
                print(f"Detected {actual_num_classes} classes from checkpoint")
        
        # Initialize model
        self.model = BiasFreeDenoisingGNN(
            num_nodes=config['num_nodes'],
            num_classes=actual_num_classes,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded: {config['model_name']}")
        print(f"Best training loss: {checkpoint['best_loss']:.4f}")
        print(f"Num nodes: {config['num_nodes']}, Num classes: {config['num_classes']}")
    
    @torch.no_grad()
    def sample_labels(self, num_samples=1):
        """
        Sample node labels from marginal distribution
        
        Returns:
            Y: [num_samples, num_nodes] node labels
        """
        Y_marginal = self.diffusion.Y_marginal
        num_nodes = self.diffusion.num_nodes
        
        Y_samples = []
        for _ in range(num_samples):
            Y = torch.multinomial(
                Y_marginal.unsqueeze(0).expand(num_nodes, -1),
                num_samples=1
            ).squeeze(-1)
            Y_samples.append(Y)
        
        return torch.stack(Y_samples) if num_samples > 1 else Y_samples[0]
    
    @torch.no_grad()
    def sample(self, num_samples=1, return_pyg=True):
        """
        Generate graphs using reverse diffusion
        
        Args:
            num_samples: Number of graphs to generate
            return_pyg: If True, return PyG Data objects
        
        Returns:
            graphs: List of generated graphs (PyG Data or edge_index)
        """
        print(f"Generating {num_samples} graphs...")
        
        generated_graphs = []
        
        for sample_idx in tqdm(range(num_samples)):
            # Sample labels
            Y = self.sample_labels().to(self.device)
            
            # Start from noise (sample from marginal)
            E_t = torch.multinomial(
                self.diffusion.E_marginal.unsqueeze(0).expand(
                    self.diffusion.num_possible_edges, -1
                ),
                num_samples=1
            ).squeeze(-1)
            
            # Reverse diffusion
            for t in reversed(range(self.diffusion.T + 1)):
                t_normalized = torch.tensor([t / self.diffusion.T], device=self.device)
                
                # Convert E_t to edge_index for model
                edge_index_t = self.diffusion.edge_vector_to_adjacency(E_t)
                
                # Predict E_0
                pred_E_logits = self.model(edge_index_t, Y, t_normalized)
                
                if t > 0:
                    # Compute posterior and sample E_{t-1}
                    prob_E_prev = self.diffusion.compute_posterior(
                        E_t, pred_E_logits, t
                    )
                    E_t = self.diffusion.sample_edge_from_prob(prob_E_prev)
                else:
                    # At t=0, take argmax
                    E_t = pred_E_logits.argmax(dim=-1)
            
            # Convert final E_0 to edge_index
            edge_index_final = self.diffusion.edge_vector_to_adjacency(E_t)
            
            if return_pyg:
                # Create PyG Data object
                graph = Data(
                    edge_index=edge_index_final.cpu(),
                    y=Y.cpu(),
                    num_nodes=self.diffusion.num_nodes
                )
                generated_graphs.append(graph)
            else:
                generated_graphs.append(edge_index_final.cpu())
        
        print(f"Generation complete!")
        return generated_graphs
    
    def save_samples(self, num_samples, output_path):
        """
        Generate and save samples
        
        Args:
            num_samples: Number of graphs to generate
            output_path: Path to save samples
        """
        graphs = self.sample(num_samples, return_pyg=True)
        
        # Save as list of (graph, metadata) tuples
        # Metadata will be empty for generated graphs
        samples = [(g, {}) for g in graphs]
        
        torch.save(samples, output_path)
        print(f"Saved {num_samples} samples to {output_path}")


def generate_from_checkpoint(checkpoint_path, num_samples, output_path=None, device='cuda'):
    """
    Convenience function to generate graphs
    
    Args:
        checkpoint_path: Path to trained model
        num_samples: Number of graphs to generate
        output_path: Where to save (if None, just return)
        device: Device to use
    
    Returns:
        generated_graphs: List of PyG Data objects
    """
    sampler = GraphSampler(checkpoint_path, device)
    graphs = sampler.sample(num_samples, return_pyg=True)
    
    if output_path is not None:
        samples = [(g, {}) for g in graphs]
        torch.save(samples, output_path)
        print(f"Saved to {output_path}")
    
    return graphs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate graphs from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of graphs to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for generated graphs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if args.output is None:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f'generated_{checkpoint_name}.pt'
    
    # Generate
    graphs = generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_path=args.output,
        device=args.device
    )
    
    # Print statistics
    print(f"\nGenerated {len(graphs)} graphs:")
    num_edges = [g.num_edges for g in graphs]
    print(f"  Edges: mean={sum(num_edges)/len(num_edges):.1f}, "
          f"min={min(num_edges)}, max={max(num_edges)}")
