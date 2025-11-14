"""
Compute graph statistics and compare with test set
"""
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def compute_graph_stats(graph):
    """
    Compute 15 graph statistics for a single graph
    
    Returns the same 15 stats as in your data format:
    [num_nodes, num_edges, density, diameter, radius, avg_degree, 
     assortativity, avg_clustering, transitivity, avg_betweenness,
     avg_closeness, avg_eigenvector_centrality, num_triangles, 
     max_clique_size, chromatic_number]
    
    Args:
        graph: PyG Data object
        
    Returns:
        stats: numpy array of shape (15,)
    """
    # Convert to NetworkX
    if isinstance(graph, Data):
        G = to_networkx(graph, to_undirected=True)
    else:
        G = graph
    
    # Remove self-loops for cleaner statistics
    G.remove_edges_from(nx.selfloop_edges(G))
    
    stats = []
    
    # 1. Number of nodes
    num_nodes = G.number_of_nodes()
    stats.append(num_nodes)
    
    # 2. Number of edges
    num_edges = G.number_of_edges()
    stats.append(num_edges)
    
    # 3. Density
    density = nx.density(G)
    stats.append(density)
    
    # 4. Diameter (handle disconnected graphs)
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            # Largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            diameter = nx.diameter(G.subgraph(largest_cc))
    except:
        diameter = 0
    stats.append(diameter)
    
    # 5. Radius (handle disconnected graphs)
    try:
        if nx.is_connected(G):
            radius = nx.radius(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            radius = nx.radius(G.subgraph(largest_cc))
    except:
        radius = 0
    stats.append(radius)
    
    # 6. Average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees) if len(degrees) > 0 else 0
    stats.append(avg_degree)
    
    # 7. Assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except:
        assortativity = 0
    stats.append(assortativity)
    
    # 8. Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    stats.append(avg_clustering)
    
    # 9. Transitivity
    transitivity = nx.transitivity(G)
    stats.append(transitivity)
    
    # 10. Average betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G)
        avg_betweenness = np.mean(list(betweenness.values()))
    except:
        avg_betweenness = 0
    stats.append(avg_betweenness)
    
    # 11. Average closeness centrality
    try:
        closeness = nx.closeness_centrality(G)
        avg_closeness = np.mean(list(closeness.values()))
    except:
        avg_closeness = 0
    stats.append(avg_closeness)
    
    # 12. Average eigenvector centrality
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        avg_eigenvector = np.mean(list(eigenvector.values()))
    except:
        avg_eigenvector = 0
    stats.append(avg_eigenvector)
    
    # 13. Number of triangles
    triangles = sum(nx.triangles(G).values()) // 3
    stats.append(triangles)
    
    # 14. Max clique size (approximate for large graphs)
    try:
        if num_nodes < 50:
            max_clique = len(max(nx.find_cliques(G), key=len))
        else:
            # Approximation for large graphs
            max_clique = max([len(c) for c in nx.find_cliques(G)])
    except:
        max_clique = 0
    stats.append(max_clique)
    
    # 15. Chromatic number (approximation via greedy coloring)
    try:
        coloring = nx.greedy_color(G)
        chromatic_number = max(coloring.values()) + 1 if coloring else 0
    except:
        chromatic_number = 0
    stats.append(chromatic_number)
    
    return np.array(stats, dtype=np.float32)


def compute_stats_for_graphs(graphs, verbose=False):
    """
    Compute statistics for a list of graphs
    
    Args:
        graphs: List of PyG Data objects
        verbose: Print progress
        
    Returns:
        stats_matrix: numpy array of shape (num_graphs, 15)
    """
    stats_list = []
    
    for i, graph in enumerate(graphs):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Computed stats for {i+1}/{len(graphs)} graphs")
        
        stats = compute_graph_stats(graph)
        stats_list.append(stats)
    
    return np.array(stats_list)


def compute_mse_with_test(generated_graphs, test_data_path, verbose=True):
    """
    Compute MSE between generated graph stats and test graph stats
    
    Args:
        generated_graphs: List of PyG Data objects (generated)
        test_data_path: Path to test.pt file
        verbose: Print progress
        
    Returns:
        mse_per_stat: MSE for each of the 15 statistics
        overall_mse: Average MSE across all statistics
        gen_stats: Stats matrix for generated graphs
        test_stats: Stats matrix for test graphs
    """
    # Load test data and extract stats
    test_data = torch.load(test_data_path, weights_only=False)
    test_stats_list = []
    
    if verbose:
        print(f"\nExtracting stats from {len(test_data)} test graphs...")
    
    for graph, stats_dict in test_data:
        test_stats_list.append(stats_dict['stats'])
    
    test_stats = np.array(test_stats_list)  # Shape: (num_test, 15)
    
    # Compute stats for generated graphs
    if verbose:
        print(f"\nComputing stats for {len(generated_graphs)} generated graphs...")
    
    gen_stats = compute_stats_for_graphs(generated_graphs, verbose=verbose)
    
    # Compute MSE
    if verbose:
        print("\nComputing MSE...")
    
    # Mean stats
    gen_mean = gen_stats.mean(axis=0)
    test_mean = test_stats.mean(axis=0)
    
    # MSE per statistic
    mse_per_stat = (gen_mean - test_mean) ** 2
    overall_mse = mse_per_stat.mean()
    
    if verbose:
        print("\n" + "="*60)
        print("Graph Statistics MSE Analysis")
        print("="*60)
        stat_names = [
            'num_nodes', 'num_edges', 'density', 'diameter', 'radius',
            'avg_degree', 'assortativity', 'avg_clustering', 'transitivity',
            'avg_betweenness', 'avg_closeness', 'avg_eigenvector_centrality',
            'num_triangles', 'max_clique_size', 'chromatic_number'
        ]
        
        for i, (name, mse) in enumerate(zip(stat_names, mse_per_stat)):
            print(f"{i+1:2d}. {name:25s}: MSE = {mse:10.6f}  "
                  f"(Gen: {gen_mean[i]:8.4f}, Test: {test_mean[i]:8.4f})")
        
        print(f"\nOverall MSE: {overall_mse:.6f}")
        print("="*60)
    
    results = {
        'mse_per_stat': mse_per_stat.tolist(),
        'overall_mse': float(overall_mse),
        'gen_mean_stats': gen_mean.tolist(),
        'test_mean_stats': test_mean.tolist(),
        'stat_names': [
            'num_nodes', 'num_edges', 'density', 'diameter', 'radius',
            'avg_degree', 'assortativity', 'avg_clustering', 'transitivity',
            'avg_betweenness', 'avg_closeness', 'avg_eigenvector_centrality',
            'num_triangles', 'max_clique_size', 'chromatic_number'
        ]
    }
    
    return results, gen_stats, test_stats
