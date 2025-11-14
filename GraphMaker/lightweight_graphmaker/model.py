"""
Bias-free GNN denoiser for graph diffusion
Predicts clean edges from noisy graph
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class BiasFreeMPLayer(MessagePassing):
    """
    Message Passing layer WITHOUT bias
    
    Args:
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__(aggr='mean')  # Mean aggregation
        
        # All linear layers without bias
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, hidden_dim] node features
            edge_index: [2, num_edges] edge connectivity
        
        Returns:
            out: [num_nodes, hidden_dim] updated features
        """
        # Add self-loops for stability
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Message passing
        out = self.propagate(edge_index, x=x)
        
        # Update with concatenation of original and aggregated
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out
    
    def message(self, x_j):
        """
        Create messages from neighbors
        
        Args:
            x_j: [num_edges, hidden_dim] neighbor features
        """
        return self.msg_mlp(x_j)


class BiasFreeMLP(nn.Module):
    """Simple MLP without bias"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class EdgePredictor(nn.Module):
    """
    Predicts edge existence for all node pairs
    WITHOUT bias in any layer
    """
    def __init__(self, node_hidden_dim, edge_hidden_dim):
        super().__init__()
        
        # Encode pairs of nodes
        self.edge_encoder = BiasFreeMLP(
            in_dim=2 * node_hidden_dim,
            hidden_dim=edge_hidden_dim,
            out_dim=2,  # Binary: no edge / edge
            num_layers=3
        )
    
    def forward(self, h_nodes, dst, src):
        """
        Args:
            h_nodes: [num_nodes, node_hidden_dim] node embeddings
            dst: [num_pairs] destination node indices
            src: [num_pairs] source node indices
        
        Returns:
            edge_logits: [num_pairs, 2] logits for each edge
        """
        # Get embeddings for source and destination nodes
        h_dst = h_nodes[dst]  # [num_pairs, node_hidden_dim]
        h_src = h_nodes[src]  # [num_pairs, node_hidden_dim]
        
        # Concatenate and predict
        h_pair = torch.cat([h_dst, h_src], dim=-1)  # [num_pairs, 2*node_hidden_dim]
        edge_logits = self.edge_encoder(h_pair)  # [num_pairs, 2]
        
        return edge_logits


class BiasFreeDenoisingGNN(nn.Module):
    """
    Complete denoising model WITHOUT bias
    
    Takes noisy graph at timestep t, predicts clean edges
    
    Args:
        num_nodes: Number of nodes
        num_classes: Number of node label classes
        hidden_dim: Hidden dimension
        num_layers: Number of MP layers
        dropout: Dropout rate
    """
    def __init__(self, num_nodes, num_classes, hidden_dim=128, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Precompute upper triangular indices for edge prediction
        dst, src = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer('dst', dst)
        self.register_buffer('src', src)
        self.num_possible_edges = len(dst)
        
        # Embeddings (no bias in embedding layers)
        self.label_embedding = nn.Embedding(num_classes, hidden_dim)
        # Note: nn.Embedding has no bias parameter by default
        
        # Time embedding (MLP without bias)
        self.time_encoder = BiasFreeMLP(
            in_dim=1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # Initial projection (no bias)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Message passing layers (all bias-free)
        self.mp_layers = nn.ModuleList([
            BiasFreeMPLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Edge predictor (bias-free)
        self.edge_predictor = EdgePredictor(hidden_dim, hidden_dim)
        
        print(f"BiasFreeDenoisingGNN initialized:")
        print(f"  Nodes: {num_nodes}, Classes: {num_classes}")
        print(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")
        print(f"  Possible edges: {self.num_possible_edges}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Bias parameters: {sum(p.numel() for n, p in self.named_parameters() if 'bias' in n)}")
    
    def forward(self, edge_index, Y, t_normalized):
        """
        Args:
            edge_index: [2, num_edges] noisy edge connectivity
            Y: [num_nodes] node labels
            t_normalized: [1] or scalar, timestep / T
        
        Returns:
            edge_logits: [num_possible_edges, 2] predicted edge logits
        """
        device = Y.device
        num_nodes = Y.size(0)
        
        # Ensure t_normalized is a tensor
        if not isinstance(t_normalized, torch.Tensor):
            t_normalized = torch.tensor([t_normalized], device=device)
        if t_normalized.dim() == 0:
            t_normalized = t_normalized.unsqueeze(0)
        
        # Encode labels
        h = self.label_embedding(Y)  # [num_nodes, hidden_dim]
        
        # Encode time
        t_emb = self.time_encoder(t_normalized.unsqueeze(-1))  # [1, hidden_dim]
        
        # Add time embedding to all nodes (broadcast)
        h = h + t_emb
        
        # Initial projection
        h = self.input_proj(h)
        h = F.relu(h)
        
        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index) + h  # Residual connection
        
        # Predict edges for all possible pairs
        edge_logits = self.edge_predictor(h, self.dst, self.src)
        
        return edge_logits
    
    def get_num_parameters(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def verify_bias_free(self):
        """Verify that model has no bias parameters"""
        bias_params = [(n, p.shape) for n, p in self.named_parameters() if 'bias' in n]
        
        if len(bias_params) > 0:
            print("WARNING: Found bias parameters:")
            for name, shape in bias_params:
                print(f"  {name}: {shape}")
            return False
        else:
            print("✓ Model is bias-free!")
            return True


if __name__ == "__main__":
    # Test the model
    num_nodes = 20
    num_classes = 3
    
    model = BiasFreeDenoisingGNN(
        num_nodes=num_nodes,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1
    )
    
    # Verify bias-free
    model.verify_bias_free()
    
    # Test forward pass
    Y = torch.randint(0, num_classes, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, 30))
    t = torch.tensor([0.5])
    
    edge_logits = model(edge_index, Y, t)
    print(f"\nOutput shape: {edge_logits.shape}")
    print(f"Expected: [{model.num_possible_edges}, 2]")
    
    # Check that we can compute loss
    target_edges = torch.randint(0, 2, (model.num_possible_edges,))
    loss = F.cross_entropy(edge_logits, target_edges)
    print(f"\nTest loss: {loss.item():.4f}")
