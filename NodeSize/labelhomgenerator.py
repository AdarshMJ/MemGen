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
    
    # One-hot encoding for label-based operations
    Z = F.one_hot(labels, num_classes=num_classes).float()

    # Use a simple stochastic-block-model-style parametrization so we can
    # exactly target expected homophily and average degree (in expectation).
    # Compute number of possible same-class and cross-class edges
    counts = [int((labels == c).sum().item()) for c in range(num_classes)]
    S_pairs = 0
    for c in range(num_classes):
        n_c = counts[c]
        S_pairs += n_c * (n_c - 1) // 2
    total_pairs = num_nodes * (num_nodes - 1) // 2
    D_pairs = total_pairs - S_pairs

    # Desired expected total number of edges
    expected_edges = float(num_nodes * avg_degree) / 2.0

    # Edge-case handling: if S_pairs or D_pairs are zero, fall back to ER sampling
    if S_pairs == 0 or D_pairs == 0:
        # Fall back: sample edges with uniform probability to meet expected_edges
        p_uniform = min(1.0, expected_edges / max(1.0, total_pairs))
        A = torch.bernoulli(torch.full((num_nodes, num_nodes), p_uniform))
        A = torch.triu(A, diagonal=1)
        A = A + A.t()
    else:
        # Compute p_in and p_out that satisfy:
        # expected_edges = p_in * S_pairs + p_out * D_pairs
        # and homophily h = (p_in * S_pairs) / expected_edges
        # => p_in = h * expected_edges / S_pairs
        # => p_out = (1-h) * expected_edges / D_pairs
        p_in = float(label_homophily) * expected_edges / float(max(1, S_pairs))
        p_out = float(1.0 - label_homophily) * expected_edges / float(max(1, D_pairs))

        # Clamp probabilities to [0,1]. If clamping occurs it's because requested
        # target homophily/degree are incompatible with the class counts; we
        # fallback to rescaling to keep ratio roughly aligned.
        if p_in > 1.0 or p_out > 1.0:
            # Try to rescale by common factor so max is 1.0
            max_p = max(p_in, p_out)
            scale = 1.0 / max_p
            p_in *= scale
            p_out *= scale

        # Build adjacency by sampling upper-triangle according to class membership
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        # Precompute class index lists
        cls_idx = {c: [i for i in range(num_nodes) if labels[i].item() == c] for c in range(num_classes)}
        # Same-class edges
        for c in range(num_classes):
            idxs = cls_idx[c]
            L = len(idxs)
            if L < 2:
                continue
            # sample upper-triangular pairs
            for i_idx in range(L):
                for j_idx in range(i_idx + 1, L):
                    u = idxs[i_idx]
                    v = idxs[j_idx]
                    if torch.rand(1).item() < p_in:
                        A[u, v] = 1.0
                        A[v, u] = 1.0
        # Cross-class edges
        for a in range(num_classes):
            for b in range(a + 1, num_classes):
                idxs_a = cls_idx[a]
                idxs_b = cls_idx[b]
                if len(idxs_a) == 0 or len(idxs_b) == 0:
                    continue
                for u in idxs_a:
                    for v in idxs_b:
                        if torch.rand(1).item() < p_out:
                            A[u, v] = 1.0
                            A[v, u] = 1.0
    
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
        try:
            iterator = tqdm(iterator, desc=f"Generating graphs (h={label_homophily:.2f}, n={num_nodes})", disable=False)
        except Exception:
            # Fallback: no progress bar if stdout is closed (BrokenPipe)
            pass
    
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
                                       dirichlet_alpha: float = None, rewiring_rate: float = 0.0,
                                       make_splits: bool = False, train_size_per_split: int = 2500,
                                       test_size: int = 100, split_seed: int = 42,
                                       split_k: int = 20, split_max_degree: int = 50,
                                       wl_validate: bool = False, wl_iters: int = 5):
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

        # Optionally create dissimilar S1/S2 splits immediately
        if make_splits:
            print("Creating S1/S2/test splits with dissimilarity objective (degree histogram clustering)...")
            S1, S2, test = make_dissimilar_splits_from_graphs(
                graphs=graphs,
                train_size=train_size_per_split,
                test_size=test_size,
                split_seed=split_seed,
                k=split_k,
                max_degree=split_max_degree,
                wl_validate=wl_validate,
                wl_iters=wl_iters
            )
            base = f"labelhomophily{h:.1f}_{num_nodes}nodes"
            s1_path = output_path / f"{base}_S1.pkl"
            s2_path = output_path / f"{base}_S2.pkl"
            test_path = output_path / f"{base}_test.pkl"
            with open(s1_path, 'wb') as f:
                pickle.dump(S1, f)
            with open(s2_path, 'wb') as f:
                pickle.dump(S2, f)
            with open(test_path, 'wb') as f:
                pickle.dump(test, f)
            print(f"Saved splits: S1 ({len(S1)}), S2 ({len(S2)}), test ({len(test)})")
        
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


# ---------- Split utilities (inlined from tools/make_dissimilar_splits.py) ----------
from typing import List, Tuple
import networkx as nx


def _data_to_degree_hist(data: Data, max_degree: int) -> np.ndarray:
    n = int(getattr(data, 'num_nodes', 0))
    if n <= 0:
        return np.zeros(max_degree + 1, dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        ei = data.edge_index.detach().cpu().numpy()
        for u, v in ei.T:
            if u < n and v < n and u != v:
                deg[u] += 1
    hist = np.bincount(np.clip(deg, 0, max_degree), minlength=max_degree + 1).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _kmeans_np(X: np.ndarray, k: int, seed: int, iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    C = X[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for j in range(C.shape[0]):
            members = X[labels == j]
            if len(members) == 0:
                C[j] = X[rng.integers(0, n)]
            else:
                C[j] = members.mean(axis=0)
    return C, labels


def _graphs_to_nx(graphs: List[Data]) -> List[nx.Graph]:
    Gs = []
    for d in graphs:
        n = int(getattr(d, 'num_nodes', 0))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if hasattr(d, 'edge_index') and d.edge_index is not None and d.edge_index.numel() > 0:
            ei = d.edge_index.detach().cpu().numpy()
            edges = set()
            for u, v in ei.T:
                if u == v:
                    continue
                a, b = (int(u), int(v)) if u < v else (int(v), int(u))
                edges.add((a, b))
            G.add_edges_from(edges)
        Gs.append(G)
    return Gs


def _wl_similarity_degree_only_nx(G1: nx.Graph, G2: nx.Graph, n_iter: int = 10) -> float:
    try:
        from grakel import WeisfeilerLehman, VertexHistogram
        from grakel.utils import graph_from_networkx
        for G in (G1, G2):
            for u in G.nodes:
                G.nodes[u]['label'] = str(int(G.degree(u)))
        graphs_pair = graph_from_networkx([G1, G2], node_labels_tag='label')
        wl_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=True, base_graph_kernel=VertexHistogram)
        K = wl_kernel.fit_transform(graphs_pair)
        return float(K[0, 1])
    except Exception:
        e1 = set(G1.edges())
        e2 = set(G2.edges())
        if not e1 and not e2:
            return 1.0
        union = len(e1 | e2)
        inter = len(e1 & e2)
        return inter / union if union > 0 else 0.0


def make_dissimilar_splits_from_graphs(graphs: List[Data], train_size: int = 2500, test_size: int = 100,
                                       split_seed: int = 42, k: int = 20, max_degree: int = 50,
                                       wl_validate: bool = False, wl_iters: int = 5):
    total = len(graphs)
    if total < test_size + 2:
        raise ValueError(f"Dataset too small for test_size={test_size}")
    rng = np.random.default_rng(split_seed)
    idx = np.arange(total)
    rng.shuffle(idx)
    graphs = [graphs[i] for i in idx]
    test = graphs[-test_size:]
    train_pool = graphs[:-test_size]

    X = np.stack([_data_to_degree_hist(g, max_degree) for g in train_pool], axis=0)
    C, labels = _kmeans_np(X, k=k, seed=split_seed, iters=12)

    # Order clusters by centroid avg-degree
    degrees = np.arange(max_degree + 1, dtype=np.float32)
    centroid_avg_deg = [float((C[j] * degrees).sum()) for j in range(C.shape[0])]
    order = np.argsort(centroid_avg_deg)
    members = [np.where(labels == j)[0].tolist() for j in order]

    # Max separation assignment
    S1_idx: List[int] = []
    S2_idx: List[int] = []
    lo, hi = 0, len(members) - 1
    while (len(S1_idx) < train_size or len(S2_idx) < train_size) and lo <= hi:
        if len(S1_idx) < train_size and lo <= hi:
            m = members[lo]
            take = m[:max(0, min(train_size - len(S1_idx), len(m)))]
            S1_idx.extend(take)
            lo += 1
        if len(S2_idx) < train_size and lo <= hi:
            m = members[hi]
            take = m[:max(0, min(train_size - len(S2_idx), len(m)))]
            S2_idx.extend(take)
            hi -= 1

    pool_remain = [i for i in range(len(train_pool)) if i not in set(S1_idx) | set(S2_idx)]
    rng.shuffle(pool_remain)
    while len(S1_idx) < train_size and pool_remain:
        S1_idx.append(pool_remain.pop())
    while len(S2_idx) < train_size and pool_remain:
        S2_idx.append(pool_remain.pop())

    S1 = [train_pool[i] for i in S1_idx]
    S2 = [train_pool[i] for i in S2_idx]

    if wl_validate:
        S1_nx = _graphs_to_nx(S1)
        S2_nx = _graphs_to_nx(S2)
        sims = []
        for _ in range(500):
            i = int(rng.integers(0, len(S1_nx)))
            j = int(rng.integers(0, len(S2_nx)))
            sims.append(_wl_similarity_degree_only_nx(S1_nx[i].copy(), S2_nx[j].copy(), n_iter=wl_iters))
        print(f"  WL(deg-only) cross-mean (iters={wl_iters}): {float(np.mean(sims)):.4f}")

    return S1, S2, test


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
    HOMOPHILY_LEVELS = [0.2]
    NUM_GRAPHS_PER_LEVEL = 5100  # 2500 S1 + 2500 S2 + 100 test
    NODE_SIZES = [500]
    NUM_CLASSES = 3
    AVG_DEGREE = 3
    OUTPUT_DIR = 'data'
    # Diversity knobs (adjust as needed; defaults preserve prior behavior if zeros/None)
    HOMOPHILY_JITTER = 0.1       # std-dev for per-graph homophily
    DEGREE_JITTER = 0.15         # log-normal jitter on avg degree
    DIRICHLET_ALPHA = 0.7        # lower => more skewed label proportions
    REWIRING_RATE = 0.1          # fraction of edges to swap per graph
    # Integrated split parameters
    MAKE_SPLITS = True
    TRAIN_SIZE_PER_SPLIT = 2500
    TEST_SIZE = 100
    SPLIT_SEED = 777
    SPLIT_K = 24
    SPLIT_MAX_DEGREE = 80
    WL_VALIDATE = True
    WL_ITERS_VALIDATE = 10
    
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
            rewiring_rate=REWIRING_RATE,
            make_splits=MAKE_SPLITS,
            train_size_per_split=TRAIN_SIZE_PER_SPLIT,
            test_size=TEST_SIZE,
            split_seed=SPLIT_SEED,
            split_k=SPLIT_K,
            split_max_degree=SPLIT_MAX_DEGREE,
            wl_validate=WL_VALIDATE,
            wl_iters=WL_ITERS_VALIDATE
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
