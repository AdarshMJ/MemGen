"""
Training script for diffusion models DF1 and DF2

DF1 trains on S1.pt (high homophily)
DF2 trains on S2.pt (low homophily)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import GraphDataset, collate_graphs, get_edge_marginals, get_label_marginals
from diffusion import GraphDiffusion
from model import BiasFreeDenoisingGNN


class DiffusionTrainer:
    """
    Trainer for graph diffusion model
    
    Args:
        data_path: Path to dataset (e.g., 'data/node_20/S1.pt')
        model_name: Name for saving checkpoints (e.g., 'DF1_n20')
        hidden_dim: Hidden dimension for GNN
        num_layers: Number of MP layers
        num_timesteps: Number of diffusion timesteps
        batch_size: Batch size
        lr: Learning rate
        num_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    def __init__(self,
                 data_path,
                 model_name,
                 hidden_dim=128,
                 num_layers=3,
                 num_timesteps=100,
                 batch_size=16,
                 lr=1e-3,
                 num_epochs=100,
                 device='cuda',
                 save_dir='checkpoints'):
        
        self.data_path = data_path
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Initializing {model_name}")
        print(f"{'='*60}")
        
        # Load dataset
        self.dataset = GraphDataset(data_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
            num_workers=0  # Set to 0 for debugging
        )
        
        # Compute marginals
        self.E_marginal = get_edge_marginals(self.dataset)
        self.Y_marginal = get_label_marginals(self.dataset)
        
        # Get actual number of classes from data
        max_label = max([g.y.max().item() for g, _ in self.dataset])
        self.actual_num_classes = max_label + 1
        print(f"Detected {self.actual_num_classes} label classes in dataset")
        
        # Initialize diffusion
        self.diffusion = GraphDiffusion(
            T=num_timesteps,
            E_marginal=self.E_marginal,
            Y_marginal=self.Y_marginal,
            num_nodes=self.dataset.num_nodes
        ).to(self.device)
        
        # Initialize model (use actual_num_classes instead of dataset.num_classes)
        self.model = BiasFreeDenoisingGNN(
            num_nodes=self.dataset.num_nodes,
            num_classes=self.actual_num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1
        ).to(self.device)
        
        # Verify bias-free
        self.model.verify_bias_free()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Training config
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Tracking
        self.train_losses = []
        self.best_loss = float('inf')
        
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"{'='*60}\n")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.dataloader, desc='Training')
        for batch_graphs, _ in pbar:
            batch_graphs = batch_graphs.to(self.device)
            
            # Process each graph in the batch separately
            # (since they have different edge structures)
            batch_loss = []
            
            for i in range(batch_graphs.num_graphs):
                # Extract single graph
                mask = batch_graphs.batch == i
                Y = batch_graphs.y[mask]
                
                # Get edges for this graph
                edge_mask = (batch_graphs.batch[batch_graphs.edge_index[0]] == i)
                graph_edges = batch_graphs.edge_index[:, edge_mask]
                
                # Remap node indices to 0-based
                node_offset = mask.nonzero(as_tuple=True)[0][0]
                graph_edges = graph_edges - node_offset
                
                # Convert to edge vector
                E = self.diffusion.adjacency_to_edge_vector(graph_edges)
                
                # Forward diffusion: corrupt edges
                E_t, t, t_normalized = self.diffusion.forward_diffusion(E)
                
                # Convert to edge_index for model
                edge_index_t = self.diffusion.edge_vector_to_adjacency(E_t)
                
                # Predict clean edges
                pred_E_logits = self.model(edge_index_t, Y, t_normalized)
                
                # Compute loss
                loss = F.cross_entropy(pred_E_logits, E)
                batch_loss.append(loss)
            
            # Average loss over batch
            total_loss = torch.stack(batch_loss).mean()
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        return sum(epoch_losses) / len(epoch_losses)
    
    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.model_name}...")
        
        for epoch in range(self.num_epochs):
            avg_loss = self.train_epoch()
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(f'{self.model_name}_best.pt')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'{self.model_name}_epoch{epoch+1}.pt')
        
        # Save final model
        self.save_checkpoint(f'{self.model_name}_final.pt')
        
        # Plot losses
        self.plot_losses()
        
        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'config': {
                'data_path': self.data_path,
                'model_name': self.model_name,
                'num_nodes': self.dataset.num_nodes,
                'num_classes': self.actual_num_classes,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': len(self.model.mp_layers),
                'num_timesteps': self.diffusion.T,
                'E_marginal': self.E_marginal.tolist(),
                'Y_marginal': self.Y_marginal.tolist()
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label=self.model_name)
        plt.xlabel('Epoch', fontsize=25)
        plt.ylabel('Loss', fontsize=25)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, f'{self.model_name}_loss.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Loss plot saved: {plot_path}")
        plt.close()


def train_model(data_path, model_name, node_size, **kwargs):
    """
    Convenience function to train a model
    
    Args:
        data_path: Path to dataset
        model_name: Model identifier (DF1 or DF2)
        node_size: Number of nodes (20, 50, 100, 500)
        **kwargs: Additional training parameters
    """
    full_model_name = f"{model_name}_n{node_size}"
    
    trainer = DiffusionTrainer(
        data_path=data_path,
        model_name=full_model_name,
        **kwargs
    )
    
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--node_size', type=int, default=20, 
                        choices=[20, 50, 100, 500],
                        help='Number of nodes')
    parser.add_argument('--split', type=str, default='S1',
                        choices=['S1', 'S2'],
                        help='Dataset split')
    parser.add_argument('--model_name', type=str, default='DF1',
                        help='Model name (DF1 or DF2)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_timesteps', type=int, default=100,
                        help='Number of diffusion timesteps')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Construct data path
    data_path = f'data/node_{args.node_size}/{args.split}.pt'
    
    # Train
    trainer = train_model(
        data_path=data_path,
        model_name=args.model_name,
        node_size=args.node_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device
    )
