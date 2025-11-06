"""
Simplified Label Homophily Graph Generator

Fast generator for synthetic graphs with controlled label homophily.
No features, no structural homophily - just structure + labels.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_networkx
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx


def make_symmetric(A):
    """Make adjacency matrix symmetric (undirected graph)"""
    A = A + A.T
    A = (A > 0).float()  # Binary adjacency
    return A


def generate_label_homophilic_graph(num_nodes, num_classes, avg_degree, 
                                    label_homophily, seed=None, n_max_nodes=100,
                                    class_probs=None, rewiring_rate=0.0):
    """
    Generate a random graph with controlled label homophily.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes in the graph
    num_classes : int
        Number of node classes/labels
    avg_degree : int
        Average node degree (target)
    label_homophily : float
        Target label homophily (0 to 1)
        - 0: random connections (no homophily)
        - 1: only same-class connections (perfect homophily)
    seed : int, optional
        Random seed for reproducibility
    n_max_nodes : int
        Maximum nodes for padding (default 100)
    
    Returns:
    --------
    data : torch_geometric.data.Data
        Graph with adjacency matrix, labels, features, and stats
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate random node labels (optionally with non-uniform class proportions)
    if class_probs is None:
        labels = torch.randint(0, num_classes, (num_nodes,))
    else:
        probs = np.asarray(class_probs, dtype=np.float64)
        probs = probs / probs.sum()
        labels_np = np.random.choice(np.arange(num_classes), size=num_nodes, p=probs)
        labels = torch.from_numpy(labels_np).long()
    
    # One-hot encoding for label-based connection probabilities
    Z = F.one_hot(labels, num_classes=num_classes).float()
    
    # Create class-to-class connection probability matrix S
    # Diagonal = probability of connecting nodes with same class
    # Off-diagonal = probability of connecting nodes with different classes
    S = torch.ones(num_classes, num_classes) * (1 - label_homophily) / (num_classes - 1)
    S.fill_diagonal_(label_homophily)
    
    # Compute connection probabilities between all node pairs
    # If nodes i and j have same class: prob = label_homophily
    # If different classes: prob = (1 - label_homophily) / (num_classes - 1)
    P = Z @ S @ Z.T  # NxN probability matrix
    
    # Scale to achieve target average degree
    # Expected degree = sum of probabilities = approx num_nodes * p_base
    # We want avg_degree, so p_base should scale P accordingly
    expected_edges = num_nodes * avg_degree / 2  # Undirected graph
    density = (2 * expected_edges) / (num_nodes * (num_nodes - 1))
    P = P * density
    P = torch.clamp(P, min=0, max=1)
    
    # Sample binary adjacency matrix from Bernoulli distribution
    A = torch.bernoulli(P)
    
    # Make symmetric (undirected) and remove self-loops
    A = make_symmetric(A)
    A.fill_diagonal_(0)
    
    # Ensure graph is connected (single component, no isolated nodes)
    import networkx as nx
    from torch_geometric.utils import to_networkx as pyg_to_nx
    
    # Convert to NetworkX to check connectivity
    temp_data = Data(edge_index=A.nonzero().t(), num_nodes=num_nodes)
    G = pyg_to_nx(temp_data, to_undirected=True)
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # If multiple components or isolated nodes, connect them
    if len(components) > 1:
        # Sort components by size (largest first)
        components = sorted(components, key=len, reverse=True)
        main_component = components[0]
        
        # Connect each smaller component to the main component
        for component in components[1:]:
            # Pick a random node from this component
            component_node = list(component)[torch.randint(len(component), (1,)).item()]
            component_label = labels[component_node]
            
            # Find a node in main component, preferring same label
            main_component_list = list(main_component)
            same_label_nodes = [n for n in main_component_list if labels[n] == component_label]
            
            if len(same_label_nodes) > 0:
                # Connect to same-label node in main component
                target_node = same_label_nodes[torch.randint(len(same_label_nodes), (1,)).item()]
            else:
                # Connect to any node in main component
                target_node = main_component_list[torch.randint(len(main_component_list), (1,)).item()]
            
            # Add edge (symmetric)
            A[component_node, target_node] = 1
            A[target_node, component_node] = 1
    
    # Also ensure no isolated nodes (degree = 0)
    degrees = A.sum(dim=1)
    isolated_nodes = (degrees == 0).nonzero(as_tuple=True)[0]
    
    if len(isolated_nodes) > 0:
        # Connect each isolated node to at least one other node
        for isolated_idx in isolated_nodes:
            isolated_label = labels[isolated_idx]
            
            # Find candidates: prefer same label
            same_label_nodes = (labels == isolated_label).nonzero(as_tuple=True)[0]
            same_label_nodes = same_label_nodes[same_label_nodes != isolated_idx]
            
            if len(same_label_nodes) > 0:
                target_idx = same_label_nodes[torch.randint(len(same_label_nodes), (1,)).item()]
            else:
                all_nodes = torch.arange(num_nodes)
                all_nodes = all_nodes[all_nodes != isolated_idx]
                target_idx = all_nodes[torch.randint(len(all_nodes), (1,)).item()]
            
            # Add edge (symmetric)
            A[isolated_idx, target_idx] = 1
            A[target_idx, isolated_idx] = 1
    
    # Optional: rewire edges to increase structural diversity (preserving degree roughly)
    if rewiring_rate > 0:
        temp_data = Data(edge_index=A.nonzero().t(), num_nodes=num_nodes)
        G_rewire = pyg_to_nx(temp_data, to_undirected=True)
        try:
            nswap = int(max(0, rewiring_rate) * max(1, G_rewire.number_of_edges()))
            if nswap > 0:
                nx.double_edge_swap(G_rewire, nswap=nswap, max_tries=nswap * 10)
                # Rebuild A from rewired graph
                A = torch.zeros_like(A)
                for u, v in G_rewire.edges():
                    A[u, v] = 1
                    A[v, u] = 1
        except Exception:
            # Fallback: keep original A if swap fails
            pass

    # Convert to edge_index format
    edge_index = A.nonzero().t()
    
    # Create dummy node features (one-hot encoded labels as features)
    # This makes VGAE work without actual semantic features
    x = F.one_hot(labels, num_classes=num_classes).float()
    
    # NOTE: We DO NOT store the padded adjacency matrix here to save space
    # Padding will be done on-the-fly during training/batching
    # Storing 1000x1000 dense matrices = 4MB per graph = 8GB for 2100 graphs!
    # Instead, we store sparse edge_index which is much more efficient
    
    # Calculate basic graph statistics (15 features: 14 basic properties + label homophily)
    import networkx as nx
    from torch_geometric.utils import to_networkx as pyg_to_nx
    
    # Convert to NetworkX for stats calculation
    temp_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = pyg_to_nx(temp_data, to_undirected=True)
    
    # Calculate basic stats
    num_edges = G.number_of_edges()
    avg_deg = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    density = nx.density(G) if num_nodes > 1 else 0
    
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
    
    # Create stats tensor (15 features)
    # [0-13]: basic graph properties, [14]: label homophily
    stats = [
        num_nodes,           # 0: number of nodes
        num_edges,           # 1: number of edges
        avg_deg,             # 2: average degree
        density,             # 3: density
        avg_clustering,      # 4: clustering coefficient
        0, 0, 0, 0, 0,      # 5-9: placeholder for other properties
        0, 0, 0, 0,         # 10-13: placeholder for other properties
        label_homophily,     # 14: label homophily (target)
    ]
    stats_tensor = torch.FloatTensor(stats).unsqueeze(0)  # Shape: (1, 15)
    
    # Ensure all tensors have exactly num_nodes (not n_max_nodes)
    assert x.shape[0] == num_nodes, f"Feature matrix has {x.shape[0]} nodes, expected {num_nodes}"
    assert labels.shape[0] == num_nodes, f"Label vector has {labels.shape[0]} nodes, expected {num_nodes}"
    assert A.shape[0] == num_nodes, f"Adjacency matrix has {A.shape[0]} nodes, expected {num_nodes}"
    
    # Create PyG Data object - store sparse edge_index instead of dense adjacency matrix
    # This reduces file size from 8GB to ~100MB for 2100 graphs
    data = Data(
        x=x.float(),                          # Node features (one-hot labels) - shape: (num_nodes, num_classes)
        edge_index=edge_index,               # Graph edges (SPARSE - efficient storage)
        stats=stats_tensor.float(),          # Graph statistics (15 features)
        y=labels,                            # Node labels - shape: (num_nodes,)
        num_nodes=num_nodes,                 # Explicit node count
        label_homophily=torch.tensor(label_homophily),
        num_classes=num_classes,
        n_max_nodes=n_max_nodes              # Store padding size for later use during training
    )
    
    # Final verification
    assert data.num_nodes == num_nodes, f"Data object has num_nodes={data.num_nodes}, expected {num_nodes}"
    assert data.x.shape[0] == num_nodes, f"Data.x has {data.x.shape[0]} rows, expected {num_nodes}"
    
    return data


def measure_label_homophily(data):
    """
    Measure actual label homophily of a graph.
    
    Returns the fraction of edges connecting nodes with the same label.
    """
    edge_index = data.edge_index
    labels = data.y
    
    # Remove self loops
    edge_index_clean = remove_self_loops(edge_index)[0]
    
    if edge_index_clean.shape[1] == 0:
        return 0.0  # No edges
    
    src_labels = labels[edge_index_clean[0]]
    tgt_labels = labels[edge_index_clean[1]]
    
    # Fraction of edges with same-class endpoints
    same_class_edges = (src_labels == tgt_labels).float()
    actual_homophily = torch.mean(same_class_edges).item()
    
    return actual_homophily


def generate_dataset(num_graphs, num_nodes, num_classes, avg_degree, 
                     label_homophily, start_seed=0, verbose=True, n_max_nodes=100,
                     homophily_jitter: float = 0.0, degree_jitter: float = 0.0,
                     dirichlet_alpha: float = None, rewiring_rate: float = 0.0):
    """
    Generate a dataset of graphs with specified label homophily.
    
    Parameters:
    -----------
    num_graphs : int
        Number of graphs to generate
    num_nodes : int
        Number of nodes per graph
    num_classes : int
        Number of node classes
    avg_degree : int
        Target average node degree
    label_homophily : float
        Target label homophily (0 to 1)
    start_seed : int
        Starting seed for reproducibility
    verbose : bool
        Print progress
    n_max_nodes : int
        Maximum nodes for padding
    
    Returns:
    --------
    graphs : list
        List of PyG Data objects
    stats : dict
        Statistics about the generated dataset
    """
    graphs = []
    actual_homophilies = []
    actual_degrees = []
    
    iterator = range(num_graphs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Generating graphs (h={label_homophily:.2f}, n={num_nodes})")
    
    for i in iterator:
        seed = start_seed + i
        # Per-graph jitter for homophily and degree
        if homophily_jitter > 0:
            h_i = float(np.clip(np.random.normal(loc=label_homophily, scale=homophily_jitter), 0.0, 1.0))
        else:
            h_i = label_homophily
        if degree_jitter > 0:
            deg_scale = float(np.exp(np.random.normal(loc=0.0, scale=degree_jitter)))
            avg_deg_i = max(1, int(round(avg_degree * deg_scale)))
        else:
            avg_deg_i = avg_degree

        # Per-graph label proportions via Dirichlet (if provided)
        if dirichlet_alpha is not None and dirichlet_alpha > 0:
            class_probs = np.random.dirichlet(alpha=np.ones(num_classes) * dirichlet_alpha)
        else:
            class_probs = None

        data = generate_label_homophilic_graph(
            num_nodes=num_nodes,
            num_classes=num_classes,
            avg_degree=avg_deg_i,
            label_homophily=h_i,
            seed=seed,
            n_max_nodes=n_max_nodes,
            class_probs=class_probs,
            rewiring_rate=rewiring_rate
        )
        
        # Measure actual properties
        actual_hom = measure_label_homophily(data)
        actual_deg = data.edge_index.shape[1] / data.num_nodes  # Total edges / num_nodes
        
        actual_homophilies.append(actual_hom)
        actual_degrees.append(actual_deg)
        
        graphs.append(data)
    
    # Compute statistics
    stats = {
        'num_graphs': num_graphs,
        'num_nodes': num_nodes,
        'num_classes': num_classes,
        'target_avg_degree': avg_degree,
        'target_label_homophily': label_homophily,
        'homophily_jitter': homophily_jitter,
        'degree_jitter': degree_jitter,
        'dirichlet_alpha': dirichlet_alpha,
        'rewiring_rate': rewiring_rate,
        'actual_label_homophily_mean': np.mean(actual_homophilies),
        'actual_label_homophily_std': np.std(actual_homophilies),
        'actual_avg_degree_mean': np.mean(actual_degrees),
        'actual_avg_degree_std': np.std(actual_degrees),
    }
    
    if verbose:
        print(f"\nDataset Statistics:")
        print(f"  Target homophily: {label_homophily:.3f}")
        print(f"  Actual homophily: {stats['actual_label_homophily_mean']:.3f} ± {stats['actual_label_homophily_std']:.3f}")
        print(f"  Target degree: {avg_degree}")
        print(f"  Actual degree: {stats['actual_avg_degree_mean']:.1f} ± {stats['actual_avg_degree_std']:.1f}")
    
    return graphs, stats


def generate_multiple_homophily_levels(homophily_levels, num_graphs_per_level, 
                                       num_nodes, num_classes=5, avg_degree=8,
                                       output_dir='data', start_seed=0, n_max_nodes=100,
                                       homophily_jitter: float = 0.0, degree_jitter: float = 0.0,
                                       dirichlet_alpha: float = None, rewiring_rate: float = 0.0):
    """
    Generate datasets for multiple label homophily levels.
    
    Parameters:
    -----------
    homophily_levels : list of float
        List of target homophily values (e.g., [0.2, 0.4, 0.6, 0.8])
    num_graphs_per_level : int
        Number of graphs to generate per homophily level
    num_nodes : int
        Number of nodes per graph
    num_classes : int
        Number of node classes
    avg_degree : int
        Target average node degree
    output_dir : str
        Directory to save generated datasets
    start_seed : int
        Starting random seed
    n_max_nodes : int
        Maximum nodes for padding
    
    Returns:
    --------
    all_results : dict
        Dictionary mapping homophily level to (graphs, stats)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for h in homophily_levels:
        print(f"\n{'='*60}")
        print(f"Generating dataset with label homophily = {h:.2f}")
        print(f"{'='*60}")
        
        graphs, stats = generate_dataset(
            num_graphs=num_graphs_per_level,
            num_nodes=num_nodes,
            num_classes=num_classes,
            avg_degree=avg_degree,
            label_homophily=h,
            start_seed=start_seed,
            verbose=True,
            n_max_nodes=n_max_nodes,
            homophily_jitter=homophily_jitter,
            degree_jitter=degree_jitter,
            dirichlet_alpha=dirichlet_alpha,
            rewiring_rate=rewiring_rate
        )
        
        # Save to file
        filename = f"labelhomophily{h:.1f}_{num_nodes}nodes_graphs.pkl"
        filepath = output_path / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(graphs, f)
        
        print(f"Saved {len(graphs)} graphs to: {filepath}")
        
        # Save stats
        stats_filename = f"labelhomophily{h:.1f}_{num_nodes}nodes_stats.txt"
        stats_filepath = output_path / stats_filename
        
        with open(stats_filepath, 'w') as f:
            f.write(f"Dataset Statistics\n")
            f.write(f"{'='*60}\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        all_results[h] = (graphs, stats)
        
        # Update seed for next level
        start_seed += num_graphs_per_level
    
    print(f"\n{'='*60}")
    print(f"All datasets generated successfully!")
    print(f"Location: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    return all_results


def visualize_graphs_by_label(graphs, num_examples=6, homophily_level=None, 
                               num_nodes=None, output_path=None):
    """
    Visualize example graphs with nodes colored by their labels.
    
    Parameters:
    -----------
    graphs : list
        List of PyG Data objects
    num_examples : int
        Number of example graphs to visualize
    homophily_level : float, optional
        Homophily level for title
    num_nodes : int, optional
        Number of nodes for title
    output_path : Path or str, optional
        If provided, save figure to this path
    """
    num_examples = min(num_examples, len(graphs))
    
    # Create color map for labels
    cmap = plt.cm.get_cmap('tab10')  # Up to 10 distinct colors
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(num_examples):
        ax = axes[idx]
        data = graphs[idx]
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Get labels
        labels = data.y.numpy()
        num_classes = len(np.unique(labels))
        
        # Assign colors based on labels
        node_colors = [cmap(labels[node]) for node in G.nodes()]
        
        # Layout
        if G.number_of_nodes() > 50:
            pos = nx.spring_layout(G, seed=42, k=0.5, iterations=30)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.0, iterations=50)
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=200, alpha=0.8, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              width=1, alpha=0.5)
        
        # Measure actual homophily
        actual_hom = measure_label_homophily(data)
        
        # Graph info
        info_text = (f"Graph {idx+1}\n"
                    f"{G.number_of_nodes()}n, {G.number_of_edges()}e\n"
                    f"Homophily: {actual_hom:.3f}")
        ax.text(0.5, -0.15, info_text, transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_axis_off()
    
    # Overall title
    title = "Label Homophily Graph Examples"
    if homophily_level is not None:
        title += f" (target h={homophily_level:.2f}"
    if num_nodes is not None:
        title += f", n={num_nodes}"
    if homophily_level is not None or num_nodes is not None:
        title += ")"
    
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    
    # Add legend for labels
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=cmap(i), markersize=10,
                                 label=f'Class {i}', markeredgecolor='black')
                      for i in range(num_classes)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=num_classes,
              fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Example usage: Generate datasets for multiple homophily levels
    
    # Configuration
    HOMOPHILY_LEVELS = [0.5]
    NUM_GRAPHS_PER_LEVEL = 5100  # 2500 S1 + 2500 S2 + 100 test
    NODE_SIZES = [5,10,20,30,40,50,100,500]
    NUM_CLASSES = 3
    AVG_DEGREE = 3
    OUTPUT_DIR = 'data'
    # Diversity knobs (adjust as needed; defaults preserve prior behavior if zeros/None)
    HOMOPHILY_JITTER = 0.05      # std-dev for per-graph h ~ N(h0, 0.05)
    DEGREE_JITTER = 0.15         # log-normal jitter on avg degree (std in log-space)
    DIRICHLET_ALPHA = 1.0        # label balance variability; None to disable
    REWIRING_RATE = 0.05         # fraction of edges to swap per graph (approx)
    
    print("Label Homophily Graph Generator")
    print("="*60)
    print(f"Homophily levels: {HOMOPHILY_LEVELS}")
    print(f"Node sizes: {NODE_SIZES}")
    print(f"Graphs per level: {NUM_GRAPHS_PER_LEVEL}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Average degree: {AVG_DEGREE}")
    print("="*60)
    
    # Create visualization directory
    vis_dir = Path(OUTPUT_DIR) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate for each node size
    for num_nodes in NODE_SIZES:
        print(f"\n\n{'#'*60}")
        print(f"# Generating graphs with {num_nodes} nodes")
        print(f"{'#'*60}")
        
        # ALWAYS pad to 500 (maximum graph size in our experiments - capped for feasibility)
        # The actual graphs will have EXACTLY num_nodes, padding is only for the adjacency matrix
        # This ensures the autoencoder always works with fixed-size 500x500 matrices
        n_max_nodes = 500
        
        results = generate_multiple_homophily_levels(
            homophily_levels=HOMOPHILY_LEVELS,
            num_graphs_per_level=NUM_GRAPHS_PER_LEVEL,
            num_nodes=num_nodes,
            num_classes=NUM_CLASSES,
            avg_degree=AVG_DEGREE,
            output_dir=OUTPUT_DIR,
            start_seed=num_nodes * 10000,  # Different seed range per node size
            n_max_nodes=n_max_nodes,
            homophily_jitter=HOMOPHILY_JITTER,
            degree_jitter=DEGREE_JITTER,
            dirichlet_alpha=DIRICHLET_ALPHA,
            rewiring_rate=REWIRING_RATE
        )
        
        # Visualize examples for each homophily level
        print(f"\n--- Creating visualizations for n={num_nodes} ---")
        for h in HOMOPHILY_LEVELS:
            graphs, stats = results[h]
            vis_path = vis_dir / f"examples_h{h:.1f}_n{num_nodes}.png"
            visualize_graphs_by_label(
                graphs=graphs,
                num_examples=6,
                homophily_level=h,
                num_nodes=num_nodes,
                output_path=vis_path
            )
    
    print("\n" + "="*60)
    print("All datasets generated!")
    print(f"Visualizations saved to: {vis_dir.absolute()}")
    print("="*60)
