"""
Discrete diffusion process for graph generation
Implements forward corruption and reverse denoising
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoiseSchedule(nn.Module):
    """
    Cosine noise schedule for diffusion
    
    Args:
        T: Number of diffusion timesteps
        s: Small constant for numerical stability
    """
    def __init__(self, T, s=0.008):
        super().__init__()
        
        self.T = T
        
        # Cosine schedule
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  # Normalize
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        
        betas = 1 - alphas
        betas = np.clip(betas, 0, 0.9999)
        alphas = 1 - betas
        
        log_alphas = np.log(alphas)
        log_alpha_bars = np.cumsum(log_alphas)
        alpha_bars = np.exp(log_alpha_bars)
        
        # Register as buffers (not parameters)
        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.register_buffer('alphas', torch.from_numpy(alphas).float())
        self.register_buffer('alpha_bars', torch.from_numpy(alpha_bars).float())
    
    def get_alpha_bar(self, t):
        """Get alpha_bar at timestep t"""
        return self.alpha_bars[t]


class MarginalTransition(nn.Module):
    """
    Marginal-based transition matrices for discrete diffusion
    
    Args:
        E_marginal: [2] tensor with [P(no edge), P(edge)]
    """
    def __init__(self, E_marginal):
        super().__init__()
        
        # Identity matrix (2x2 for binary edge existence)
        I_E = torch.eye(2)
        
        # Marginal matrix: each row is the marginal distribution
        # Shape: (2, 2) where each row is [P(no edge), P(edge)]
        m_E = E_marginal.unsqueeze(0).expand(2, -1).clone()
        
        self.register_buffer('I_E', I_E)
        self.register_buffer('m_E', m_E)
    
    def get_Q_bar_E(self, alpha_bar_t):
        """
        Compute transition matrix Q_bar_t for edges
        
        Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * m
        
        Args:
            alpha_bar_t: Scalar or [1] tensor
        
        Returns:
            Q_bar_t_E: [2, 2] transition matrix
        """
        Q_bar_t_E = alpha_bar_t * self.I_E + (1 - alpha_bar_t) * self.m_E
        return Q_bar_t_E


class GraphDiffusion(nn.Module):
    """
    Complete diffusion process for graphs
    Focus on edge generation only (not node features)
    
    Args:
        T: Number of diffusion timesteps
        E_marginal: [2] tensor with edge marginals
        Y_marginal: [num_classes] tensor with label marginals
        num_nodes: Number of nodes in graphs
    """
    def __init__(self, T, E_marginal, Y_marginal, num_nodes):
        super().__init__()
        
        self.T = T
        self.num_nodes = num_nodes
        self.num_classes_Y = len(Y_marginal)
        
        self.noise_schedule = NoiseSchedule(T)
        self.transition = MarginalTransition(E_marginal)
        
        self.register_buffer('E_marginal', E_marginal)
        self.register_buffer('Y_marginal', Y_marginal)
        
        # For upper triangular edge indexing
        dst, src = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer('dst', dst)
        self.register_buffer('src', src)
        self.num_possible_edges = len(dst)
        
        print(f"GraphDiffusion initialized: T={T}, nodes={num_nodes}, "
              f"possible_edges={self.num_possible_edges}")
    
    def adjacency_to_edge_vector(self, edge_index, num_nodes=None):
        """
        Convert edge_index to binary edge existence vector
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            num_nodes: Number of nodes (if None, use self.num_nodes)
        
        Returns:
            E: [num_possible_edges] binary vector (1 if edge exists)
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Undirected
        
        # Extract upper triangular
        E = adj[self.dst, self.src]
        return E.long()
    
    def edge_vector_to_adjacency(self, E):
        """
        Convert binary edge vector to edge_index
        
        Args:
            E: [num_possible_edges] binary vector
        
        Returns:
            edge_index: [2, num_edges] edge connectivity
        """
        # Get edges where E=1
        edge_mask = E == 1
        dst_edges = self.dst[edge_mask]
        src_edges = self.src[edge_mask]
        
        # Make bidirectional
        edge_index = torch.stack([
            torch.cat([dst_edges, src_edges]),
            torch.cat([src_edges, dst_edges])
        ], dim=0)
        
        return edge_index
    
    def sample_edge_from_prob(self, prob_E):
        """
        Sample edge existence from probability distribution
        
        Args:
            prob_E: [num_possible_edges, 2] probabilities
        
        Returns:
            E: [num_possible_edges] binary vector
        """
        E = prob_E.multinomial(1).squeeze(-1)
        return E
    
    def forward_diffusion(self, E, t=None):
        """
        Apply forward diffusion: corrupt edges
        
        Args:
            E: [num_possible_edges] binary edge vector
            t: Timestep (if None, sample randomly)
        
        Returns:
            E_t: [num_possible_edges] corrupted edges
            t: Timestep used
            t_normalized: t/T for model input
        """
        if t is None:
            t = torch.randint(0, self.T + 1, (1,), device=E.device).item()
        
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t)
        
        # Get transition matrix
        Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t)  # [2, 2]
        
        # Convert E to one-hot: [num_possible_edges, 2]
        E_one_hot = F.one_hot(E, num_classes=2).float()
        
        # Compute probabilities: E_one_hot @ Q_bar_t_E
        prob_E = E_one_hot @ Q_bar_t_E  # [num_possible_edges, 2]
        
        # Sample corrupted edges
        E_t = self.sample_edge_from_prob(prob_E)
        
        t_normalized = torch.tensor([t / self.T], device=E.device)
        
        return E_t, t, t_normalized
    
    def compute_posterior(self, E_t, pred_E_logits, t):
        """
        Compute posterior q(E_{t-1} | E_t, pred_E_0) for reverse diffusion
        
        Args:
            E_t: [num_possible_edges] edges at timestep t
            pred_E_logits: [num_possible_edges, 2] predicted logits for E_0
            t: Current timestep
        
        Returns:
            prob_E_prev: [num_possible_edges, 2] probabilities for t-1
        """
        if t == 0:
            # At t=0, just return predicted probabilities
            return F.softmax(pred_E_logits, dim=-1)
        
        s = t - 1
        alpha_bar_s = self.noise_schedule.get_alpha_bar(s)
        alpha_t = self.noise_schedule.alphas[t]
        
        Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s)  # [2, 2]
        Q_t_E = self.transition.get_Q_bar_E(alpha_t)  # [2, 2]
        
        # E_t one-hot: [num_possible_edges, 2]
        E_t_one_hot = F.one_hot(E_t, num_classes=2).float()
        
        # Predicted E_0 probabilities
        pred_E_prob = F.softmax(pred_E_logits, dim=-1)  # [num_possible_edges, 2]
        
        # Compute posterior using Bayes rule
        # p(E_{t-1} | E_t, E_0) ∝ q(E_t | E_{t-1}) * q(E_{t-1} | E_0)
        
        # Left term: E_t @ Q_t^T -> [num_possible_edges, 2]
        left_term = E_t_one_hot @ Q_t_E.T
        left_term = left_term.unsqueeze(-2)  # [num_possible_edges, 1, 2]
        
        # Right term: [2, 2] -> [1, 2, 2]
        right_term = Q_bar_s_E.unsqueeze(0)
        
        # Numerator: [num_possible_edges, 2, 2]
        numerator = left_term * right_term
        
        # For each possible E_0 value, weight by predicted probability
        # pred_E_prob: [num_possible_edges, 2]
        # We want to marginalize over E_0
        prob_E_prev = (pred_E_prob.unsqueeze(-1) * numerator).sum(dim=-2)
        
        # Normalize
        prob_E_prev = prob_E_prev / (prob_E_prev.sum(dim=-1, keepdim=True) + 1e-10)
        
        return prob_E_prev
    
    @torch.no_grad()
    def reverse_diffusion(self, denoiser, Y, num_samples=1):
        """
        Sample graphs using reverse diffusion
        
        Args:
            denoiser: Model that predicts E_0 from E_t
            Y: [num_nodes] node labels
            num_samples: Number of graphs to generate
        
        Returns:
            generated_edge_indices: List of [2, num_edges] edge_index tensors
        """
        device = Y.device
        
        generated_graphs = []
        
        for _ in range(num_samples):
            # Start from noise (sample from marginal)
            E_t = torch.multinomial(
                self.E_marginal.unsqueeze(0).expand(self.num_possible_edges, -1),
                num_samples=1
            ).squeeze(-1)
            
            # Reverse diffusion
            for t in reversed(range(self.T + 1)):
                t_normalized = torch.tensor([t / self.T], device=device)
                
                # Convert E_t to edge_index for denoiser
                edge_index_t = self.edge_vector_to_adjacency(E_t)
                
                # Predict E_0
                pred_E_logits = denoiser(edge_index_t, Y, t_normalized)
                
                if t > 0:
                    # Compute posterior and sample E_{t-1}
                    prob_E_prev = self.compute_posterior(E_t, pred_E_logits, t)
                    E_t = self.sample_edge_from_prob(prob_E_prev)
                else:
                    # At t=0, take argmax
                    E_t = pred_E_logits.argmax(dim=-1)
            
            # Convert final E_0 to edge_index
            edge_index_final = self.edge_vector_to_adjacency(E_t)
            generated_graphs.append(edge_index_final)
        
        return generated_graphs


if __name__ == "__main__":
    # Test diffusion
    E_marginal = torch.tensor([0.8, 0.2])  # 20% edges
    Y_marginal = torch.tensor([0.3, 0.3, 0.4])  # 3 classes
    num_nodes = 20
    T = 100
    
    diffusion = GraphDiffusion(T, E_marginal, Y_marginal, num_nodes)
    
    # Create a test graph
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    E = diffusion.adjacency_to_edge_vector(edge_index)
    
    print(f"Original E: {E.sum()} edges out of {diffusion.num_possible_edges}")
    
    # Test forward diffusion
    E_t, t, t_norm = diffusion.forward_diffusion(E, t=50)
    print(f"Corrupted E at t={t}: {E_t.sum()} edges")
    
    # Test edge conversion
    edge_index_t = diffusion.edge_vector_to_adjacency(E_t)
    print(f"Edge index shape: {edge_index_t.shape}")
