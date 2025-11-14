"""
Weisfeiler-Lehman graph kernel for measuring similarity
"""
import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict, Counter
import networkx as nx
from grakel import GraphKernel


def pyg_to_networkx(pyg_graph):
    """
    Convert PyG Data to NetworkX graph
    
    Args:
        pyg_graph: PyG Data object
    
    Returns:
        G: NetworkX graph with node labels
    """
    G = nx.Graph()
    
    # Add nodes with labels
    for i in range(pyg_graph.num_nodes):
        label = pyg_graph.y[i].item() if pyg_graph.y is not None else 0
        G.add_node(i, label=str(label))
    
    # Add edges
    edge_index = pyg_graph.edge_index
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u < v:  # Undirected, only add once
            G.add_edge(u, v)
    
    return G


def compute_wl_kernel(graphs1, graphs2, n_iter=5, return_kernel_matrix=False):
    """
    Compute Weisfeiler-Lehman kernel similarity between two sets of graphs
    
    Args:
        graphs1: List of PyG Data objects or NetworkX graphs
        graphs2: List of PyG Data objects or NetworkX graphs
        n_iter: Number of WL iterations
        return_kernel_matrix: If True, return full kernel matrix
    
    Returns:
        similarity_matrix: [len(graphs1), len(graphs2)] similarity matrix
        mean_similarity: Average similarity
        K12 (optional): Raw kernel matrix if return_kernel_matrix=True
    """
    # Convert to NetworkX if needed
    if isinstance(graphs1[0], Data):
        graphs1 = [pyg_to_networkx(g) for g in graphs1]
    if isinstance(graphs2[0], Data):
        graphs2 = [pyg_to_networkx(g) for g in graphs2]
    
    # Convert to grakel format (adjacency matrix + node labels)
    def nx_to_grakel_format(G):
        """Convert NetworkX graph to grakel format"""
        # Get adjacency dict
        adj = {i: list(G.neighbors(i)) for i in G.nodes()}
        # Get node labels
        node_labels = {i: G.nodes[i].get('label', '0') for i in G.nodes()}
        return [adj, node_labels]
    
    grakel_graphs1 = [nx_to_grakel_format(g) for g in graphs1]
    grakel_graphs2 = [nx_to_grakel_format(g) for g in graphs2]
    
    # Compute WL kernel
    print(f"Computing WL kernel (n_iter={n_iter})...")
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": n_iter}, 
                             {"name": "vertex_histogram"}])
    
    # Fit on all graphs
    all_graphs = grakel_graphs1 + grakel_graphs2
    gk.fit(all_graphs)
    
    # Transform to get kernel matrix
    K = gk.transform(all_graphs)
    
    # Extract submatrices
    n1 = len(graphs1)
    n2 = len(graphs2)
    
    # K11: graphs1 vs graphs1
    K11 = K[:n1, :n1]
    # K22: graphs2 vs graphs2
    K22 = K[n1:, n1:]
    # K12: graphs1 vs graphs2
    K12 = K[:n1, n1:]
    
    # Normalize kernel values to [0, 1] similarity
    # K(G1, G2) / sqrt(K(G1, G1) * K(G2, G2))
    K11_diag = np.diag(K11).reshape(-1, 1)
    K22_diag = np.diag(K22).reshape(1, -1)
    
    # Avoid division by zero
    denom = np.sqrt(K11_diag @ K22_diag) + 1e-10
    similarity_matrix = K12 / denom
    
    mean_similarity = similarity_matrix.mean()
    
    if return_kernel_matrix:
        return similarity_matrix, mean_similarity, K12
    return similarity_matrix, mean_similarity


def compute_wl_similarity(graphs1, graphs2, n_iter=5, return_matrix=False):
    """
    Simplified function that returns mean WL similarity
    
    Args:
        graphs1: List of graphs
        graphs2: List of graphs
        n_iter: Number of WL iterations
        return_matrix: If True, also return similarity matrix
    
    Returns:
        mean_similarity: Average similarity score [0, 1]
        similarity_matrix (optional): [len(graphs1), len(graphs2)] matrix if return_matrix=True
    """
    sim_matrix, mean_sim = compute_wl_kernel(graphs1, graphs2, n_iter)
    if return_matrix:
        return mean_sim, sim_matrix
    return mean_sim


def find_closest_matches(gen_graphs, train_graphs, n_iter=5):
    """
    For each generated graph, find the closest match in training set
    
    Args:
        gen_graphs: List of generated graphs
        train_graphs: List of training graphs
        n_iter: Number of WL iterations
    
    Returns:
        closest_indices: [len(gen_graphs)] indices of closest training graphs
        closest_similarities: [len(gen_graphs)] similarity scores
        similarity_matrix: [len(gen_graphs), len(train_graphs)] full matrix
    """
    print(f"Finding closest matches for {len(gen_graphs)} generated graphs...")
    similarity_matrix, _ = compute_wl_kernel(gen_graphs, train_graphs, n_iter)
    
    # For each generated graph, find training graph with highest similarity
    closest_indices = similarity_matrix.argmax(axis=1)
    closest_similarities = similarity_matrix.max(axis=1)
    
    return closest_indices, closest_similarities, similarity_matrix


def evaluate_memorization_vs_generalization(gen1_graphs, gen2_graphs, 
                                           s1_graphs, s2_graphs,
                                           n_iter=5,
                                           output_dir=None):
    """
    Evaluate memorization vs generalization
    
    Computes:
    - WLSim(Gen1, S1): How similar Gen1 is to its training set
    - WLSim(Gen2, S2): How similar Gen2 is to its training set
    - WLSim(Gen1, Gen2): How similar the two generated sets are
    - Finds closest matches for visualization
    
    Args:
        gen1_graphs: Graphs generated by DF1
        gen2_graphs: Graphs generated by DF2
        s1_graphs: Training set S1
        s2_graphs: Training set S2
        n_iter: WL iterations
        output_dir: Directory to save visualizations (if None, skip visualizations)
    
    Returns:
        results: Dict with similarity scores and distributions
    """
    print("\n" + "="*60)
    print("Evaluating Memorization vs Generalization")
    print("="*60)
    
    results = {}
    
    # Find closest matches: Gen1 vs S1
    print("\nFinding closest matches: Gen1 vs S1...")
    closest_idx_g1_s1, closest_sim_g1_s1, sim_matrix_g1_s1 = find_closest_matches(
        gen1_graphs, s1_graphs, n_iter
    )
    results['closest_indices_Gen1_S1'] = closest_idx_g1_s1.tolist()
    results['closest_similarities_Gen1_S1'] = closest_sim_g1_s1.tolist()
    results['WLSim_Gen1_S1'] = float(np.mean(closest_sim_g1_s1))
    print(f"Mean WLSim(Gen1, S1) = {results['WLSim_Gen1_S1']:.4f}")
    
    # Find closest matches: Gen2 vs S2
    print("\nFinding closest matches: Gen2 vs S2...")
    closest_idx_g2_s2, closest_sim_g2_s2, sim_matrix_g2_s2 = find_closest_matches(
        gen2_graphs, s2_graphs, n_iter
    )
    results['closest_indices_Gen2_S2'] = closest_idx_g2_s2.tolist()
    results['closest_similarities_Gen2_S2'] = closest_sim_g2_s2.tolist()
    results['WLSim_Gen2_S2'] = float(np.mean(closest_sim_g2_s2))
    print(f"Mean WLSim(Gen2, S2) = {results['WLSim_Gen2_S2']:.4f}")
    
    # Compute Gen1 vs Gen2 similarity
    print("\nComputing WLSim(Gen1, Gen2)...")
    sim_matrix_g1_g2, mean_g1_g2 = compute_wl_kernel(gen1_graphs, gen2_graphs, n_iter)
    results['WLSim_Gen1_Gen2'] = float(mean_g1_g2)
    results['similarity_matrix_Gen1_Gen2'] = sim_matrix_g1_g2.tolist()
    print(f"Mean WLSim(Gen1, Gen2) = {results['WLSim_Gen1_Gen2']:.4f}")
    
    # Additional: Cross-contamination checks
    print("\nComputing cross-contamination metrics...")
    results['WLSim_Gen1_S2'] = compute_wl_similarity(gen1_graphs, s2_graphs, n_iter)
    results['WLSim_Gen2_S1'] = compute_wl_similarity(gen2_graphs, s1_graphs, n_iter)
    print(f"WLSim(Gen1, S2) = {results['WLSim_Gen1_S2']:.4f}")
    print(f"WLSim(Gen2, S1) = {results['WLSim_Gen2_S1']:.4f}")
    
    # Compute memorization vs generalization score
    avg_memorization = (results['WLSim_Gen1_S1'] + results['WLSim_Gen2_S2']) / 2
    generalization = results['WLSim_Gen1_Gen2']
    
    results['avg_memorization'] = avg_memorization
    results['generalization'] = generalization
    results['gen_vs_mem_ratio'] = generalization / (avg_memorization + 1e-10)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Average Memorization: {avg_memorization:.4f}")
    print(f"Generalization: {generalization:.4f}")
    print(f"Gen/Mem Ratio: {results['gen_vs_mem_ratio']:.4f}")
    
    if results['gen_vs_mem_ratio'] > 1.2:
        print("\n✓ Models appear to GENERALIZE (learn same distribution)")
    elif results['gen_vs_mem_ratio'] < 0.8:
        print("\n✓ Models appear to MEMORIZE (training sets distinct)")
    else:
        print("\n? Intermediate regime")
    
    print("="*60 + "\n")
    
    # Create visualizations if output_dir provided
    if output_dir is not None:
        print("Creating visualizations...")
        try:
            from visualize import create_comprehensive_visualization
            
            create_comprehensive_visualization(
                gen1_graphs=gen1_graphs,
                gen2_graphs=gen2_graphs,
                s1_graphs=s1_graphs,
                s2_graphs=s2_graphs,
                closest_indices_g1_s1=closest_idx_g1_s1,
                closest_indices_g2_s2=closest_idx_g2_s2,
                closest_sims_g1_s1=closest_sim_g1_s1,
                closest_sims_g2_s2=closest_sim_g2_s2,
                sim_matrix_g1_g2=sim_matrix_g1_g2,
                output_dir=output_dir
            )
        except ImportError:
            print("Warning: Could not import visualize module, skipping visualizations")
    
    return results


if __name__ == "__main__":
    # Test WL kernel computation
    print("Testing WL kernel...")
    
    # Create some test graphs
    def create_test_graph(num_nodes, num_edges):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randint(0, 3, (num_nodes,))
        return Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
    
    graphs1 = [create_test_graph(20, 25) for _ in range(5)]
    graphs2 = [create_test_graph(20, 30) for _ in range(5)]
    
    sim = compute_wl_similarity(graphs1, graphs2, n_iter=3)
    print(f"WL Similarity: {sim:.4f}")
    
    # Test full evaluation
    print("\nTesting full evaluation...")
    results = evaluate_memorization_vs_generalization(
        graphs1, graphs2, graphs1, graphs2, n_iter=2
    )
