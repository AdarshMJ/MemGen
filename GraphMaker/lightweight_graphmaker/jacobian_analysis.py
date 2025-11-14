"""
Jacobian analysis for graph denoiser
Adapted from linear_approx.py for image denoisers

Computes the Jacobian of the graph denoising function to analyze:
- Local linearity of the denoiser
- Dimensionality of the denoising manifold
- PCA of denoised outputs around a clean graph
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def calc_jacobian_graph(edge_vector, Y, t_normalized, model, diffusion):
    """
    Calculate full Jacobian of the graph denoiser with respect to input edge vector
    
    For images: J[i,j] = ∂(denoiser_output[i]) / ∂(input[j])
    For graphs: J[i,j] = ∂(denoiser_logits[i]) / ∂(edge_vector[j])
    
    Args:
        edge_vector: [num_possible_edges] edge values (should be continuous for gradient)
        Y: [num_nodes] node labels
        t_normalized: scalar or [1], timestep / T
        model: BiasFreeDenoisingGNN model
        diffusion: GraphDiffusion object
    
    Returns:
        jacob: [num_possible_edges, num_possible_edges] Jacobian matrix
    """
    # Prepare model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Ensure edge_vector requires grad
    if not edge_vector.requires_grad:
        edge_vector.requires_grad = True
    
    # For gradient computation, we need to use a differentiable path
    # We'll use the edge_index from the binarized version but compute gradients through edge_vector
    # This approximates the Jacobian around the current edge configuration
    with torch.no_grad():
        edge_index = diffusion.edge_vector_to_adjacency(edge_vector.long())
    
    # Get prediction - the model internally uses the edge_index structure
    # but we need to track how changes in edge_vector affect the output
    # We'll use a workaround: compute model output and manually construct Jacobian
    
    # Since the model uses discrete edge_index, true analytical Jacobian is not available
    # We approximate using finite differences or use the logits directly
    pred_logits = model(edge_index, Y, t_normalized)  # [num_possible_edges, 2]
    
    # Get predicted edge probabilities (class 1 = edge exists)
    pred_probs = F.softmax(pred_logits, dim=-1)[:, 1]  # [num_possible_edges]
    
    # Since we can't backprop through discrete edge_index construction,
    # we approximate the Jacobian by flipping edges (0→1 or 1→0)
    jacob = []
    num_edges = edge_vector.size(0)
    
    print(f"Computing Jacobian for {num_edges} edges using edge perturbations...")
    with torch.no_grad():
        # Baseline prediction
        edge_index = diffusion.edge_vector_to_adjacency(edge_vector.long())
        pred_logits = model(edge_index, Y, t_normalized)
        baseline_probs = F.softmax(pred_logits, dim=-1)[:, 1]
        
        for i in tqdm(range(num_edges), desc="Jacobian rows"):
            row = []
            for j in range(num_edges):
                # Flip edge j (0→1 or 1→0)
                edge_vector_perturbed = edge_vector.clone()
                edge_vector_perturbed[j] = 1.0 - edge_vector_perturbed[j]
                
                # Get new prediction
                edge_index_perturbed = diffusion.edge_vector_to_adjacency(edge_vector_perturbed.long())
                pred_logits_perturbed = model(edge_index_perturbed, Y, t_normalized)
                pred_probs_perturbed = F.softmax(pred_logits_perturbed, dim=-1)[:, 1]
                
                # Discrete derivative (effect of flipping edge j on output i)
                partial = (pred_probs_perturbed[i] - baseline_probs[i]).item()
                row.append(partial)
            
            jacob.append(torch.tensor(row, device=edge_vector.device))
    
    jacob = torch.stack(jacob)  # [num_possible_edges, num_possible_edges]
    
    return jacob


def calc_jacobian_row_graph(edge_vector, Y, t_normalized, model, diffusion, row_idx):
    """
    Calculate a single row of the Jacobian (faster for spot checks)
    
    Args:
        edge_vector: [num_possible_edges] edge values
        Y: [num_nodes] node labels
        t_normalized: scalar or [1], timestep / T
        model: BiasFreeDenoisingGNN model
        diffusion: GraphDiffusion object
        row_idx: Which output edge to compute gradient for
    
    Returns:
        jacob_row: [num_possible_edges] single row of Jacobian
    """
    # Prepare model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Use edge flipping (discrete perturbation)
    num_edges = edge_vector.size(0)
    jacob_row = []
    
    with torch.no_grad():
        # Baseline
        edge_index = diffusion.edge_vector_to_adjacency(edge_vector.long())
        pred_logits = model(edge_index, Y, t_normalized)
        pred_probs = F.softmax(pred_logits, dim=-1)[:, 1]
        baseline_val = pred_probs[row_idx]
        
        # Compute gradient via edge flipping
        for j in range(num_edges):
            edge_vector_perturbed = edge_vector.clone()
            edge_vector_perturbed[j] = 1.0 - edge_vector_perturbed[j]  # Flip edge
            
            edge_index_perturbed = diffusion.edge_vector_to_adjacency(edge_vector_perturbed.long())
            pred_logits_perturbed = model(edge_index_perturbed, Y, t_normalized)
            pred_probs_perturbed = F.softmax(pred_logits_perturbed, dim=-1)[:, 1]
            
            partial = (pred_probs_perturbed[row_idx] - baseline_val).item()
            jacob_row.append(partial)
    
    return torch.tensor(jacob_row, device=edge_vector.device)


def approx_subspace_proj(U_sub, S_sub, V_sub, x):
    """
    Project onto the tangent plane defined by SVD components
    
    Args:
        U_sub: Left singular vectors
        S_sub: Singular values
        V_sub: Right singular vectors
        x: Vector to project
    
    Returns:
        projection: Projected vector reshaped to x.shape
    """
    temp = torch.matmul(V_sub, x.flatten())
    temp = torch.matmul(torch.diag(S_sub), temp)
    return torch.matmul(U_sub, temp).reshape(x.shape)


def pca_denoised_graphs(clean_edge_vector, Y, model, diffusion, 
                        noise_levels, num_samples, device='cpu'):
    """
    Perform PCA on denoised graphs around a clean graph
    
    Analogous to pca_denoised for images:
    1. Take a clean graph
    2. Add noise at different levels (different timesteps)
    3. Denoise the noisy graphs
    4. Compute PCA of the denoised outputs
    
    Args:
        clean_edge_vector: [num_possible_edges] clean edge vector
        Y: [num_nodes] node labels
        model: BiasFreeDenoisingGNN model
        diffusion: GraphDiffusion object
        noise_levels: List of noise timesteps (e.g., [20, 40, 60, 80, 100])
        num_samples: Number of noisy samples per noise level
        device: Device to run on
    
    Returns:
        eigenvalues: Dict {t: eigenvalues} for each noise level
        eigenvectors: Dict {t: eigenvectors} for each noise level
        denoised_graphs: Dict {t: denoised_edge_vectors} for each noise level
    """
    eigenvalues = {}
    eigenvectors = {}
    denoised_graphs = {}
    
    model.eval()
    
    print(f"\nPerforming PCA analysis with {num_samples} samples per noise level")
    print(f"Noise levels (timesteps): {noise_levels}")
    
    for t in noise_levels:
        print(f"\n=== Noise level t={t} ===")
        
        # Generate multiple noisy versions
        noisy_samples = []
        t_normalized = torch.tensor([t / diffusion.T], device=device)
        
        for i in tqdm(range(num_samples), desc=f"Generating samples at t={t}"):
            # Add noise using forward diffusion
            E_t, _, _ = diffusion.forward_diffusion(clean_edge_vector, t=t)
            noisy_samples.append(E_t)
        
        noisy_batch = torch.stack(noisy_samples)  # [num_samples, num_possible_edges]
        
        # Denoise all samples
        denoised_batch = []
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc=f"Denoising at t={t}"):
                E_t = noisy_batch[i]
                edge_index_t = diffusion.edge_vector_to_adjacency(E_t)
                
                # Get denoiser prediction
                pred_logits = model(edge_index_t, Y, t_normalized)
                pred_probs = F.softmax(pred_logits, dim=-1)[:, 1]  # [num_possible_edges]
                
                denoised_batch.append(pred_probs)
        
        denoised_batch = torch.stack(denoised_batch)  # [num_samples, num_possible_edges]
        denoised_graphs[t] = denoised_batch
        
        # Center the denoised outputs
        mean_denoised = denoised_batch.mean(dim=0, keepdim=True)
        centered = denoised_batch - mean_denoised
        
        # Compute covariance matrix
        print(f"Computing covariance matrix...")
        cov = torch.cov(centered.T)  # [num_possible_edges, num_possible_edges]
        
        # Eigen decomposition
        print(f"Computing eigendecomposition...")
        L, Q = torch.linalg.eig(cov)
        eigenvalues[t] = torch.real(L)
        eigenvectors[t] = torch.real(Q)
        
        # Sort by eigenvalue magnitude
        sorted_indices = torch.argsort(eigenvalues[t], descending=True)
        eigenvalues[t] = eigenvalues[t][sorted_indices]
        eigenvectors[t] = eigenvectors[t][:, sorted_indices]
        
        print(f"Top 5 eigenvalues: {eigenvalues[t][:5].cpu().numpy()}")
        print(f"Explained variance (top 10): {eigenvalues[t][:10].sum() / eigenvalues[t].sum():.4f}")
    
    return eigenvalues, eigenvectors, denoised_graphs


def compute_effective_rank(eigenvalues, threshold=0.99):
    """
    Compute effective rank: number of eigenvalues needed to explain threshold variance
    
    Args:
        eigenvalues: [num_components] sorted eigenvalues
        threshold: Variance threshold (default 0.99 = 99%)
    
    Returns:
        effective_rank: Number of components explaining threshold variance
        cumsum: Cumulative explained variance
    """
    total_var = eigenvalues.sum()
    cumsum = torch.cumsum(eigenvalues, dim=0) / total_var
    effective_rank = (cumsum < threshold).sum().item() + 1
    
    return effective_rank, cumsum


def analyze_jacobian_properties(jacob):
    """
    Analyze properties of the Jacobian matrix
    
    Args:
        jacob: [n, n] Jacobian matrix
    
    Returns:
        analysis: Dict with various metrics
    """
    analysis = {}
    
    # Spectral properties
    print("\nComputing SVD...")
    U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
    
    analysis['singular_values'] = S
    analysis['condition_number'] = (S[0] / S[-1]).item()
    analysis['rank'] = torch.linalg.matrix_rank(jacob).item()
    analysis['frobenius_norm'] = torch.linalg.norm(jacob, ord='fro').item()
    
    # Effective rank at different thresholds
    for threshold in [0.90, 0.95, 0.99]:
        eff_rank, _ = compute_effective_rank(S**2, threshold)
        analysis[f'effective_rank_{int(threshold*100)}'] = eff_rank
    
    # Diagonality measure (how close to identity)
    diag_vals = torch.diag(jacob)
    off_diag_norm = torch.linalg.norm(jacob - torch.diag(diag_vals), ord='fro')
    total_norm = torch.linalg.norm(jacob, ord='fro')
    analysis['off_diagonal_ratio'] = (off_diag_norm / total_norm).item()
    
    print(f"Jacobian analysis:")
    print(f"  Rank: {analysis['rank']}")
    print(f"  Condition number: {analysis['condition_number']:.2e}")
    print(f"  Effective rank (99%): {analysis['effective_rank_99']}")
    print(f"  Off-diagonal ratio: {analysis['off_diagonal_ratio']:.4f}")
    
    return analysis


def traj_projections_graph(edge_vectors, Y, t_normalized, model, diffusion):
    """
    Project trajectory of intermediate denoised graphs
    
    Args:
        edge_vectors: [num_steps, num_possible_edges] trajectory
        Y: [num_nodes] node labels
        t_normalized: scalar or [1], timestep / T
        model: BiasFreeDenoisingGNN model
        diffusion: GraphDiffusion object
    
    Returns:
        projected: [num_steps, num_possible_edges] denoised trajectory
    """
    projected = []
    
    with torch.no_grad():
        for E_t in edge_vectors:
            edge_index_t = diffusion.edge_vector_to_adjacency(E_t)
            pred_logits = model(edge_index_t, Y, t_normalized)
            pred_probs = F.softmax(pred_logits, dim=-1)[:, 1]
            projected.append(pred_probs)
    
    return torch.stack(projected)
