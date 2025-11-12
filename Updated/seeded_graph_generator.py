#!/usr/bin/env python3
"""Fast seeded synthetic graph generator."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy as sp
import torch
import networkx as nx
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops
from utils import gen_stats


@dataclass
class SeedStats:
    name: str
    class_probs: np.ndarray
    pair_probs: np.ndarray
    avg_degree: float
    num_classes: int


def _coalesce_edges(edge_index: torch.Tensor) -> np.ndarray:
    edge_index, _ = remove_self_loops(edge_index)
    edges = set()
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return np.array(list(edges), dtype=np.int64)


def load_seed_stats(name: str, root: str) -> SeedStats:
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    edges = _coalesce_edges(data.edge_index)
    labels = data.y.cpu().numpy()
    num_nodes = int(data.num_nodes)
    num_classes = int(labels.max()) + 1

    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_probs = class_counts / class_counts.sum()

    pair_counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for u, v in edges:
        cu, cv = labels[u], labels[v]
        pair_counts[cu, cv] += 1.0
        if cu != cv:
            pair_counts[cv, cu] += 1.0

    if pair_counts.sum() == 0:
        pair_probs = np.full((num_classes, num_classes), 1.0 / (num_classes**2))
    else:
        pair_probs = pair_counts / pair_counts.sum()

    avg_degree = 2.0 * len(edges) / num_nodes if num_nodes else 0.0

    return SeedStats(
        name=name,
        class_probs=class_probs,
        pair_probs=pair_probs,
        avg_degree=avg_degree,
        num_classes=num_classes,
    )


def adjust_pair_probs(seed: SeedStats, target_hom: float, jitter: float) -> np.ndarray:
    probs = seed.pair_probs.copy()
    diag_mask = np.eye(seed.num_classes, dtype=bool)
    diag_sum = probs[diag_mask].sum()
    off_sum = probs[~diag_mask].sum()

    if target_hom <= 0.0:
        probs[diag_mask] = 1e-8
        off_value = 1.0 / (seed.num_classes * (seed.num_classes - 1))
        probs[~diag_mask] = off_value
    elif target_hom >= 1.0:
        probs[diag_mask] = 1.0 / seed.num_classes
        probs[~diag_mask] = 1e-8
    else:
        if diag_sum > 0:
            probs[diag_mask] *= target_hom / diag_sum
        else:
            probs[diag_mask] = target_hom / seed.num_classes
        if off_sum > 0:
            probs[~diag_mask] *= (1.0 - target_hom) / off_sum
        else:
            fill = (1.0 - target_hom) / max(seed.num_classes * (seed.num_classes - 1), 1)
            probs[~diag_mask] = fill

    probs = np.clip(probs, 1e-8, None)
    probs = 0.5 * (probs + probs.T)
    probs /= probs.sum()

    if jitter > 0:
        noise = np.random.dirichlet(np.ones(probs.size)).reshape(probs.shape)
        probs = (1.0 - jitter) * probs + jitter * noise
        probs = 0.5 * (probs + probs.T)
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

    return probs


def sample_labels(class_probs: np.ndarray, n: int) -> np.ndarray:
    labels = np.random.choice(len(class_probs), size=n, p=class_probs)
    return labels


def sample_edge_budget(avg_degree: float, n: int) -> int:
    mean_edges = max(avg_degree, 1.0) * n / 2.0
    noise = np.random.normal(loc=0.0, scale=max(1.0, 0.05 * mean_edges))
    total = max(1, int(round(mean_edges + noise)))
    return total


def allocate_edge_counts(pair_probs: np.ndarray, total_edges: int) -> np.ndarray:
    num_classes = pair_probs.shape[0]
    upper = np.triu(pair_probs)
    upper /= upper.sum()
    flat = upper[np.triu_indices(num_classes)]
    counts = np.random.multinomial(total_edges, flat)
    result = np.zeros_like(pair_probs, dtype=np.int64)
    idx = np.triu_indices(num_classes)
    result[idx] = counts
    result[(idx[1], idx[0])] = counts
    return result


def build_edge_set(labels: np.ndarray, pair_counts: np.ndarray) -> List[Tuple[int, int]]:
    num_classes = pair_counts.shape[0]
    class_nodes = [np.where(labels == c)[0] for c in range(num_classes)]
    edges: set[Tuple[int, int]] = set()

    for i in range(num_classes):
        nodes_i = class_nodes[i]
        if nodes_i.size == 0:
            continue
        for j in range(i, num_classes):
            need = int(pair_counts[i, j])
            if need <= 0:
                continue
            nodes_j = class_nodes[j]
            if nodes_j.size == 0:
                continue
            attempts = 0
            max_attempts = max(need * 5, 50)
            while need > 0 and attempts < max_attempts:
                u = int(np.random.choice(nodes_i))
                v = int(np.random.choice(nodes_j))
                if i == j and u == v:
                    attempts += 1
                    continue
                edge = (u, v) if u < v else (v, u)
                if edge in edges:
                    attempts += 1
                    continue
                edges.add(edge)
                need -= 1
            if need > 0:
                continue

    return list(edges)


def realised_label_hom(edges: List[Tuple[int, int]], labels: np.ndarray) -> float:
    if not edges:
        return 0.0
    same = sum(1 for u, v in edges if labels[u] == labels[v])
    return same / len(edges)


def ensure_connected(edges: List[Tuple[int, int]], n: int, labels: np.ndarray, target_hom: float) -> List[Tuple[int, int]]:
    """Ensure graph is connected by adding minimal edges between components.
    
    Uses Union-Find to identify connected components, then connects them by adding
    edges that respect the target homophily as much as possible.
    """
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Build initial components from existing edges
    edge_set = set(edges)
    for u, v in edges:
        union(u, v)
    
    # Find all components
    components = {}
    for node in range(n):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)
    
    # If already connected, return original edges
    if len(components) == 1:
        return edges
    
    # Connect components with minimal edges
    component_list = list(components.values())
    new_edges = list(edges)
    
    # Connect each component to the first one
    for i in range(1, len(component_list)):
        comp_a = component_list[0]
        comp_b = component_list[i]
        
        # Try to find an edge that respects homophily
        best_edge = None
        best_score = -1.0
        
        for u in comp_a:
            for v in comp_b:
                edge = (u, v) if u < v else (v, u)
                if edge in edge_set:
                    continue
                
                # Score: prefer same-label edges if target_hom > 0.5, else different-label
                same_label = (labels[u] == labels[v])
                if target_hom > 0.5:
                    score = 1.0 if same_label else 0.0
                else:
                    score = 0.0 if same_label else 1.0
                
                if score > best_score or best_edge is None:
                    best_score = score
                    best_edge = edge
        
        if best_edge:
            new_edges.append(best_edge)
            edge_set.add(best_edge)
            # Update union-find
            union(best_edge[0], best_edge[1])
    
    return new_edges


def compute_node_features_with_bfs(edges: List[Tuple[int, int]], n: int, spectral_dim: int = 10, n_max_nodes: int = 100) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute node features using BFS ordering, degree, and Laplacian eigenvectors.
    
    This mirrors the approach in main.py:
    1. Build networkx graph
    2. Apply BFS ordering from highest-degree node in each component
    3. Compute normalized Laplacian eigenvectors
    4. Create features: [normalized_degree, eigenvector_1, ..., eigenvector_k]
    
    Args:
        edges: List of (u, v) edge tuples
        n: Number of nodes
        spectral_dim: Number of Laplacian eigenvectors to include
        n_max_nodes: Maximum nodes for normalization
        
    Returns:
        node_features: [n, spectral_dim+1] tensor
        bfs_node_list: Array mapping new_idx -> original_idx
    """
    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])
    
    # Get connected components, sorted by size
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    
    # BFS ordering from highest-degree node in each component
    node_list_bfs = []
    for component in CGs:
        node_degree_list = [(n, d) for n, d in component.degree()]
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
        bfs_tree = nx.bfs_tree(component, source=degree_sequence[0][0])
        node_list_bfs += list(bfs_tree.nodes())
    
    # Build adjacency matrix in BFS order
    adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
    adj = torch.from_numpy(adj_bfs).float()
    
    # Compute normalized Laplacian
    diags = np.sum(adj_bfs, axis=0)
    diags = np.squeeze(np.asarray(diags))
    D = sparse.diags(diags).toarray()
    L = D - adj_bfs
    
    with np.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sparse.diags(diags_sqrt).toarray()
    L = np.linalg.multi_dot((DH, L, DH))
    L = torch.from_numpy(L).float()
    
    # Compute eigenvalues and eigenvectors
    eigval, eigvecs = torch.linalg.eigh(L)
    eigval = torch.real(eigval)
    eigvecs = torch.real(eigvecs)
    idx = torch.argsort(eigval)
    eigvecs = eigvecs[:, idx]
    
    # Build node features: [normalized_degree, eigenvector_1, ..., eigenvector_k]
    x = torch.zeros(n, spectral_dim + 1)
    # Normalized degree (using n_max_nodes for consistency with main.py)
    x[:, 0] = torch.mm(adj, torch.ones(n, 1))[:, 0] / (n_max_nodes - 1)
    # Eigenvectors (take up to spectral_dim)
    mn = min(n, spectral_dim)
    x[:, 1:mn+1] = eigvecs[:, :mn]
    
    return x, np.array(node_list_bfs, dtype=np.int64)


def generate_single_graph(
    templates: Sequence[SeedStats],
    n: int,
    target_hom: float,
    jitter: float,
    spectral_dim: int = 10,
    n_max_nodes: int = 100,
) -> Tuple[Data, Dict[str, float]]:
    for attempt in range(50):
        template = random.choice(templates)
        labels = sample_labels(template.class_probs, n)
        pair_probs = adjust_pair_probs(template, target_hom, jitter)
        total_edges = sample_edge_budget(template.avg_degree, n)
        pair_counts = allocate_edge_counts(pair_probs, total_edges)
        edges = build_edge_set(labels, pair_counts)
        if not edges:
            continue

        # Ensure connectivity
        edges = ensure_connected(edges, n, labels, target_hom)

        # Compute node features with BFS ordering
        node_features, bfs_node_list = compute_node_features_with_bfs(
            edges, n, spectral_dim=spectral_dim, n_max_nodes=n_max_nodes
        )
        
        # Reorder edges according to BFS ordering
        # Create mapping from original -> BFS index
        orig_to_bfs = {int(orig_idx): bfs_idx for bfs_idx, orig_idx in enumerate(bfs_node_list)}
        edges_reordered = [(orig_to_bfs[u], orig_to_bfs[v]) for u, v in edges]
        
        # Reorder labels according to BFS
        labels_reordered = labels[bfs_node_list]

        edge_array = np.array(edges_reordered, dtype=np.int64)
        edge_index = torch.from_numpy(edge_array.T.copy()).long()
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.from_numpy(labels_reordered).long()
        )
        data.num_nodes = n

        realised_hom = realised_label_hom(edges, labels)
        avg_degree = 2.0 * len(edges) / n if n else 0.0
        meta = {
            "seed": template.name,
            "target_label_hom": target_hom,
            "realised_label_hom": realised_hom,
            "avg_degree": avg_degree,
            "num_edges": len(edges),
        }

        # Compute and attach precomputed stats (using project's gen_stats helper)
        try:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])
            stats_array = gen_stats(G)
            # Ensure a 1D numpy array and coerce to length 15 (pad/truncate)
            stats_array = np.asarray(stats_array).ravel().astype(float)
            target_len = 15
            if stats_array.size < target_len:
                stats_array = np.pad(stats_array, (0, target_len - stats_array.size), constant_values=0.0)
            elif stats_array.size > target_len:
                stats_array = stats_array[:target_len]
            meta['stats'] = stats_array
        except Exception:
            # If stats computation fails, leave it out and let downstream compute
            meta['stats'] = None

        return data, meta

    raise RuntimeError("Graph sampling failed after multiple retries.")


def compute_wl_similarity(graphs: List[Tuple[Data, Dict[str, float]]], n_iter: int = 3) -> float:
    """Compute average pairwise WL similarity within a set of graphs.
    
    Args:
        graphs: List of (Data, meta) tuples
        n_iter: Number of WL iterations
        
    Returns:
        Average pairwise WL similarity (0-1, higher = more similar)
    """
    try:
        from grakel import Graph as GrakelGraph
        from grakel.kernels import WeisfeilerLehman, VertexHistogram
        
        # Convert to grakel format
        grakel_graphs = []
        for data, _ in graphs:
            edges = data.edge_index.t().tolist()
            n = int(data.num_nodes)
            edge_dict = {i: [] for i in range(n)}
            for u, v in edges:
                u, v = int(u), int(v)
                if u != v:
                    edge_dict[u].append(v)
                    edge_dict[v].append(u)
            node_labels = {i: 0 for i in range(n)}  # uniform labels for structure
            grakel_graphs.append(GrakelGraph(edge_dict, node_labels=node_labels))
        
        # Compute WL kernel
        wl_kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
        K = wl_kernel.fit_transform(grakel_graphs)
        
        # Average off-diagonal similarities
        n_graphs = len(graphs)
        if n_graphs < 2:
            return 0.0
        similarities = []
        for i in range(n_graphs):
            for j in range(i + 1, n_graphs):
                similarities.append(K[i, j])
        
        return float(np.mean(similarities)) if similarities else 0.0
    except Exception as e:
        print(f"Warning: WL similarity computation failed: {e}")
        return 0.0


def compute_pairwise_diversity(graphs: List[Tuple[Data, Dict[str, float]]], metric: str = 'homophily') -> float:
    """Compute diversity within a set of graphs.
    
    Args:
        graphs: List of (Data, meta) tuples
        metric: 'homophily' (std of homophily) or 'wl' (1 - WL similarity)
        
    Returns:
        Diversity score (higher = more diverse)
    """
    if metric == 'homophily':
        homs = [meta.get('realised_label_hom', 0.5) for _, meta in graphs]
        return float(np.std(homs))
    elif metric == 'wl':
        sim = compute_wl_similarity(graphs, n_iter=3)
        return 1.0 - sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_cross_diversity(graphs1: List[Tuple[Data, Dict[str, float]]], 
                           graphs2: List[Tuple[Data, Dict[str, float]]], 
                           metric: str = 'homophily') -> float:
    """Compute diversity between two sets of graphs.
    
    Args:
        graphs1, graphs2: Lists of (Data, meta) tuples
        metric: 'homophily' (difference of means) or 'wl' (1 - cross-WL similarity)
        
    Returns:
        Cross-diversity score (higher = more diverse)
    """
    if metric == 'homophily':
        homs1 = [meta.get('realised_label_hom', 0.5) for _, meta in graphs1]
        homs2 = [meta.get('realised_label_hom', 0.5) for _, meta in graphs2]
        return abs(float(np.mean(homs1)) - float(np.mean(homs2)))
    elif metric == 'wl':
        # Compute cross-similarity
        try:
            from grakel import Graph as GrakelGraph
            from grakel.kernels import WeisfeilerLehman, VertexHistogram
            
            # Convert both sets to grakel
            all_graphs = graphs1 + graphs2
            grakel_graphs = []
            for data, _ in all_graphs:
                edges = data.edge_index.t().tolist()
                n = int(data.num_nodes)
                edge_dict = {i: [] for i in range(n)}
                for u, v in edges:
                    u, v = int(u), int(v)
                    if u != v:
                        edge_dict[u].append(v)
                        edge_dict[v].append(u)
                node_labels = {i: 0 for i in range(n)}
                grakel_graphs.append(GrakelGraph(edge_dict, node_labels=node_labels))
            
            # Compute WL kernel
            wl_kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
            K = wl_kernel.fit_transform(grakel_graphs)
            
            # Average cross similarities
            n1 = len(graphs1)
            n2 = len(graphs2)
            cross_sims = []
            for i in range(n1):
                for j in range(n2):
                    cross_sims.append(K[i, n1 + j])
            
            avg_cross_sim = float(np.mean(cross_sims)) if cross_sims else 0.0
            return 1.0 - avg_cross_sim
        except Exception as e:
            print(f"Warning: Cross WL diversity computation failed: {e}")
            return 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_dataset_splits(
    templates: Sequence[SeedStats],
    node_size: int,
    total_graphs: int,
    train_per_set: int,
    test_count: int,
    target_hom: float,
    jitter: float,
    spectral_dim: int = 10,
    n_max_nodes: int = 100,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
    """Legacy function: generates splits with identical parameters (NO diversity).
    
    For diverse splits, use generate_dataset_splits_diverse() instead.
    """
    required = 2 * train_per_set + test_count
    if total_graphs < required:
        raise ValueError(f"Need at least {required} graphs, received {total_graphs}")

    graphs: List[Tuple[Data, Dict[str, float]]] = []
    while len(graphs) < total_graphs:
        graph, meta = generate_single_graph(
            templates=templates,
            n=node_size,
            target_hom=target_hom,
            jitter=jitter,
            spectral_dim=spectral_dim,
            n_max_nodes=n_max_nodes,
        )
        graphs.append((graph, meta))
        if len(graphs) % 200 == 0:
            print(f"  generated {len(graphs)} / {total_graphs} graphs for n={node_size}")

    random.shuffle(graphs)
    s1 = graphs[:train_per_set]
    s2 = graphs[train_per_set : 2 * train_per_set]
    test = graphs[2 * train_per_set : 2 * train_per_set + test_count]
    extra = graphs[2 * train_per_set + test_count :]

    return {"S1": s1, "S2": s2, "test": test, "extra": extra}


def generate_dataset_splits_diverse(
    templates: Sequence[SeedStats],
    node_size: int,
    train_per_set: int,
    test_count: int,
    s1_config: Dict,
    s2_config: Dict,
    test_config: Dict,
    spectral_dim: int = 10,
    n_max_nodes: int = 100,
    validate_diversity: bool = True,
    diversity_margin: float = 0.15,
    validation_sample_size: int = 100,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
    """Generate dataset splits with DIVERSE S1 and S2 configurations.
    
    Args:
        templates: List of seed graph statistics (e.g., Cora, CiteSeer)
        node_size: Number of nodes per graph
        train_per_set: Number of graphs per training split (S1/S2)
        test_count: Number of test graphs
        s1_config: Config for S1 split, e.g. {'target_hom_range': [0.65, 0.95], 'jitter': 0.1, 'templates': [0]}
        s2_config: Config for S2 split, e.g. {'target_hom_range': [0.15, 0.45], 'jitter': 0.1, 'templates': [1]}
        test_config: Config for test split, e.g. {'target_hom_range': [0.4, 0.6], 'jitter': 0.1, 'templates': None}
        spectral_dim: Number of Laplacian eigenvectors for node features
        n_max_nodes: Max nodes for degree normalization
        validate_diversity: Whether to validate diversity constraints during generation
        diversity_margin: Minimum required gap: cross_diversity > within_diversity + margin
        validation_sample_size: Sample size for diversity validation
        
    Returns:
        Dictionary with splits: {'S1': [...], 'S2': [...], 'test': [...]}
        
    Config format:
        - target_hom_range: [low, high] homophily range (will sample uniformly)
        - jitter: Dirichlet noise level for pair probabilities
        - templates: List of template indices to use (None = all templates)
    """
    print(f"\n=== Generating DIVERSE splits for n={node_size} ===")
    print(f"S1 config: {s1_config}")
    print(f"S2 config: {s2_config}")
    print(f"Test config: {test_config}")
    print(f"Validation: {'ENABLED' if validate_diversity else 'DISABLED'} (margin={diversity_margin})")
    
    def generate_split(config: Dict, count: int, split_name: str) -> List[Tuple[Data, Dict[str, float]]]:
        """Generate graphs for a single split with given configuration."""
        hom_range = config.get('target_hom_range', [0.4, 0.6])
        jitter = config.get('jitter', 0.1)
        template_indices = config.get('templates', None)
        
        # Filter templates if specified
        if template_indices is not None:
            split_templates = [templates[i] for i in template_indices if i < len(templates)]
        else:
            split_templates = list(templates)
        
        if not split_templates:
            raise ValueError(f"No valid templates for {split_name}")
        
        graphs = []
        while len(graphs) < count:
            # Sample homophily uniformly from range
            target_hom = random.uniform(hom_range[0], hom_range[1])
            
            graph, meta = generate_single_graph(
                templates=split_templates,
                n=node_size,
                target_hom=target_hom,
                jitter=jitter,
                spectral_dim=spectral_dim,
                n_max_nodes=n_max_nodes,
            )
            graphs.append((graph, meta))
            
            if len(graphs) % 200 == 0:
                print(f"  {split_name}: generated {len(graphs)} / {count} graphs")
        
        return graphs
    
    # Generate S1
    print(f"\nGenerating S1 ({train_per_set} graphs)...")
    s1_graphs = generate_split(s1_config, train_per_set, "S1")
    
    # Generate S2
    print(f"\nGenerating S2 ({train_per_set} graphs)...")
    s2_graphs = generate_split(s2_config, train_per_set, "S2")
    
    # Generate test
    print(f"\nGenerating test ({test_count} graphs)...")
    test_graphs = generate_split(test_config, test_count, "test")
    
    # Validate diversity constraints
    if validate_diversity:
        print(f"\n=== Validating diversity constraints ===")
        
        # Sample for validation (full set too expensive)
        val_size = min(validation_sample_size, train_per_set)
        s1_sample = random.sample(s1_graphs, val_size)
        s2_sample = random.sample(s2_graphs, val_size)
        
        # Compute diversity metrics
        print("Computing within-split diversity (homophily std)...")
        s1_within_hom = compute_pairwise_diversity(s1_sample, metric='homophily')
        s2_within_hom = compute_pairwise_diversity(s2_sample, metric='homophily')
        
        print("Computing cross-split diversity (homophily difference)...")
        cross_hom = compute_cross_diversity(s1_sample, s2_sample, metric='homophily')
        
        avg_within_hom = (s1_within_hom + s2_within_hom) / 2.0
        margin_hom = cross_hom - avg_within_hom
        
        print(f"\n--- Homophily Diversity ---")
        print(f"S1 within (std): {s1_within_hom:.4f}")
        print(f"S2 within (std): {s2_within_hom:.4f}")
        print(f"Avg within: {avg_within_hom:.4f}")
        print(f"Cross (|mean diff|): {cross_hom:.4f}")
        print(f"Margin: {margin_hom:.4f} (target: >{diversity_margin:.2f})")
        
        if margin_hom < diversity_margin:
            print(f"⚠️  WARNING: Homophily margin {margin_hom:.4f} < {diversity_margin:.2f}")
            print(f"    Consider adjusting target_hom_range configs")
        else:
            print(f"✓ Homophily diversity constraint satisfied!")
        
        # Optional: WL diversity (expensive, only for small samples)
        if val_size <= 50:
            print("\nComputing WL diversity (may take a while)...")
            try:
                s1_within_wl = compute_pairwise_diversity(s1_sample, metric='wl')
                s2_within_wl = compute_pairwise_diversity(s2_sample, metric='wl')
                cross_wl = compute_cross_diversity(s1_sample, s2_sample, metric='wl')
                
                avg_within_wl = (s1_within_wl + s2_within_wl) / 2.0
                margin_wl = cross_wl - avg_within_wl
                
                print(f"\n--- WL Diversity ---")
                print(f"S1 within (1-sim): {s1_within_wl:.4f}")
                print(f"S2 within (1-sim): {s2_within_wl:.4f}")
                print(f"Avg within: {avg_within_wl:.4f}")
                print(f"Cross (1-sim): {cross_wl:.4f}")
                print(f"Margin: {margin_wl:.4f}")
                
                if margin_wl < 0.05:  # Lower threshold for WL
                    print(f"⚠️  WARNING: WL margin {margin_wl:.4f} is low")
                else:
                    print(f"✓ WL diversity looks good!")
            except Exception as e:
                print(f"WL diversity computation failed (non-critical): {e}")
        
        print("\n" + "="*50)
    
    return {"S1": s1_graphs, "S2": s2_graphs, "test": test_graphs}


def save_split(output_dir: str, node_size: int, split: Dict[str, List[Tuple[Data, Dict[str, float]]]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.join(output_dir, f"node_{node_size}")
    os.makedirs(base_dir, exist_ok=True)

    for split_name, entries in split.items():
        data_path = os.path.join(base_dir, f"{split_name}.pt")
        # Entries are tuples (Data, meta) where meta may contain 'stats' (numpy array)
        torch.save(entries, data_path)

    summary = {}
    for split_name, entries in split.items():
        homs = [meta["realised_label_hom"] for _, meta in entries]
        summary[split_name] = {
            "count": len(entries),
            "mean_label_hom": float(np.mean(homs)) if homs else None,
            "std_label_hom": float(np.std(homs)) if homs else None,
        }

    with open(os.path.join(base_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


def visualize_examples(output_dir: str, node_size: int, n_examples: int = 3, layout: str = 'spring') -> None:
    """Load saved split for node_size and render a few example graphs colored by labels.

    This writes a PNG image into the split directory called `examples_node_{n}.png`.
    """
    base_dir = os.path.join(output_dir, f"node_{node_size}")
    s1_path = os.path.join(base_dir, 'S1.pt')
    if not os.path.exists(s1_path):
        print(f"No S1 split found at {s1_path}")
        return

    entries = torch.load(s1_path, map_location='cpu', weights_only=False)
    if len(entries) == 0:
        print("No entries in S1 split to visualize")
        return

    # Sample up to n_examples graphs
    take = min(n_examples, len(entries))
    chosen = random.sample(entries, take)

    # Use a compact figure size so lines remain visible even when preview downscales
    fig, axes = plt.subplots(1, take, figsize=(4 * take, 4))
    if take == 1:
        axes = [axes]

    for ax, (data, meta) in zip(axes, chosen):
        # Convert to networkx for drawing
        G = nx.Graph()
        n = int(data.num_nodes)
        G.add_nodes_from(range(n))
        edges = data.edge_index.t().tolist()
        G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])

        labels = None
        if hasattr(data, 'y') and data.y is not None:
            labels = data.y.cpu().numpy().astype(int).tolist()
        elif meta is not None and 'labels' in meta:
            labels = meta['labels']

        if labels is None:
            # fallback: color all nodes the same
            node_colors = ['#3498db'] * n
        else:
            # map unique labels to colors
            uniq = sorted(set(labels))
            cmap = plt.get_cmap('Set1')  # More vibrant colors
            color_map = {lab: cmap(i % 9) for i, lab in enumerate(uniq)}
            node_colors = [color_map[int(l)] for l in labels]

        # Choose layout - use different strategies for sparse vs dense graphs
        avg_degree = 2 * G.number_of_edges() / n if n > 0 else 0
        
        try:
            if avg_degree < 3.0 or layout == 'kamada_kawai':
                # For sparse graphs, kamada_kawai works much better
                pos = nx.kamada_kawai_layout(G, scale=1.0)
            elif layout == 'circular':
                pos = nx.circular_layout(G, scale=1.0)
            else:
                # Spring layout with very aggressive parameters for sparse graphs
                k_param = max(0.5, 2.0/np.sqrt(n))  # Much higher k for more spacing
                pos = nx.spring_layout(G, k=k_param, iterations=200, seed=42, scale=1.0)
            
            # Check if layout collapsed (all positions too close)
            if pos:
                # Preserve node->position mapping while normalizing
                node_order = list(G.nodes())
                coords = np.array([pos[node] for node in node_order], dtype=float)
                # Normalize to [-1, 1]
                mins = coords.min(axis=0)
                maxs = coords.max(axis=0)
                spans = np.maximum(maxs - mins, 1e-8)
                coords = (coords - mins) / spans  # [0,1]
                coords = 2.0 * coords - 1.0       # [-1,1]
                pos = {node: (float(coords[i, 0]), float(coords[i, 1])) for i, node in enumerate(node_order)}

                # If still nearly-degenerate, force a circular layout
                if float(np.std(coords)) < 1e-3:
                    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    R = 1.0
                    pos = {node: (float(R * np.cos(theta[i])), float(R * np.sin(theta[i]))) for i, node in enumerate(node_order)}
        except Exception as e:
            # Fallback to circular if layout fails
            print(f"  Layout failed ({e}), using circular fallback")
            pos = nx.circular_layout(G, scale=1.0)

        # Add tiny jitter to avoid accidental overlap from identical coords
        jitter = 0.02
        pos = {k: (v[0] + np.random.uniform(-jitter, jitter), v[1] + np.random.uniform(-jitter, jitter)) for k, v in pos.items()}

        # Draw edges FIRST so nodes appear on top
        # Use thick, dark edges so they survive Preview.app downscaling
        edge_width = max(6.0, 160.0 / max(n, 1))  # >=6pt, thicker for tiny graphs
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.95, width=edge_width, edge_color='black',
            style='solid'
        )

        # Moderate node size so edges remain visible
        node_sz = max(250, min(700, 8000 // max(n, 1)))
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sz, ax=ax,
            edgecolors='black', linewidths=1.2, alpha=0.95
        )

        # Add node labels for small graphs
        if n <= 30:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

        # Add title with key metrics
        title = f"{meta.get('seed','?')} n={n} e={meta.get('num_edges',0)} hom={meta.get('realised_label_hom',0):.2f}"
        ax.set_title(title, fontsize=14, pad=10, weight='bold')
        ax.set_axis_off()

        # Set equal aspect ratio and fixed limits to ensure visibility
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    out_path = os.path.join(base_dir, f"examples_node_{node_size}.png")
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved examples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast seeded synthetic graph dataset generator")
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory for generated splits")
    parser.add_argument("--node-sizes", type=int, nargs="+", default=[100], help="Node counts to generate")
    parser.add_argument("--total-per-size", type=int, default=5100, help="Graphs to sample per node size (legacy mode only)")
    parser.add_argument("--train-per-set", type=int, default=2500, help="Graphs per training subset")
    parser.add_argument("--test-count", type=int, default=100, help="Graphs reserved for evaluation")
    parser.add_argument("--target-hom", type=float, default=0.5, help="Desired label homophily (legacy mode only)")
    parser.add_argument("--pair-jitter", type=float, default=0.1, help="Dirichlet noise level for pair probabilities")
    parser.add_argument("--planetoid-root", type=str, default="./planetoid", help="Planetoid dataset root directory")
    parser.add_argument("--visualize", action="store_true", help="Generate example visualizations after creating splits")
    parser.add_argument("--viz-examples", type=int, default=3, help="Number of example graphs to visualize (default: 3)")
    parser.add_argument("--spectral-dim", type=int, default=10, help="Number of Laplacian eigenvectors to use as node features (default: 10)")
    parser.add_argument("--n-max-nodes", type=int, default=100, help="Maximum number of nodes for degree normalization (default: 100)")
    
    # Diverse split generation mode
    parser.add_argument("--diverse", action="store_true", help="Enable diverse split generation (S1 and S2 have different configs)")
    parser.add_argument("--s1-hom-range", type=float, nargs=2, default=[0.65, 0.95], help="S1 homophily range [low, high] (diverse mode)")
    parser.add_argument("--s2-hom-range", type=float, nargs=2, default=[0.15, 0.45], help="S2 homophily range [low, high] (diverse mode)")
    parser.add_argument("--test-hom-range", type=float, nargs=2, default=[0.4, 0.6], help="Test homophily range [low, high] (diverse mode)")
    parser.add_argument("--s1-templates", type=int, nargs="*", default=None, help="Template indices for S1 (diverse mode, None=all)")
    parser.add_argument("--s2-templates", type=int, nargs="*", default=None, help="Template indices for S2 (diverse mode, None=all)")
    parser.add_argument("--test-templates", type=int, nargs="*", default=None, help="Template indices for test (diverse mode, None=all)")
    parser.add_argument("--validate-diversity", action="store_true", help="Validate diversity constraints during generation (diverse mode)")
    parser.add_argument("--diversity-margin", type=float, default=0.15, help="Minimum cross-within diversity gap (diverse mode)")
    parser.add_argument("--validation-sample-size", type=int, default=100, help="Sample size for diversity validation (diverse mode)")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)

    templates = [
        load_seed_stats("Cora", args.planetoid_root),
        load_seed_stats("CiteSeer", args.planetoid_root),
    ]

    if args.diverse:
        # DIVERSE MODE: S1 and S2 have different configurations
        print("\n" + "="*70)
        print("DIVERSE SPLIT GENERATION MODE ENABLED")
        print("="*70)
        
        s1_config = {
            'target_hom_range': args.s1_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.s1_templates,
        }
        s2_config = {
            'target_hom_range': args.s2_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.s2_templates,
        }
        test_config = {
            'target_hom_range': args.test_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.test_templates,
        }
        
        for node_size in args.node_sizes:
            print(f"\n{'='*70}")
            print(f"Generating DIVERSE splits for node size {node_size}...")
            print(f"{'='*70}")
            
            split = generate_dataset_splits_diverse(
                templates=templates,
                node_size=node_size,
                train_per_set=args.train_per_set,
                test_count=args.test_count,
                s1_config=s1_config,
                s2_config=s2_config,
                test_config=test_config,
                spectral_dim=args.spectral_dim,
                n_max_nodes=args.n_max_nodes,
                validate_diversity=args.validate_diversity,
                diversity_margin=args.diversity_margin,
                validation_sample_size=args.validation_sample_size,
            )
            
            save_split(args.output_dir, node_size, split)
            print(f"\n✓ Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')
    else:
        # LEGACY MODE: Identical parameters for S1 and S2
        print("\n" + "="*70)
        print("LEGACY MODE: Generating splits with IDENTICAL parameters")
        print("WARNING: This may cause S1↔S2 convergence at large n")
        print("Use --diverse flag for diverse split generation")
        print("="*70)
        
        for node_size in args.node_sizes:
            print(f"\nGenerating node size {node_size}...")
            split = generate_dataset_splits(
                templates=templates,
                node_size=node_size,
                total_graphs=args.total_per_size,
                train_per_set=args.train_per_set,
                test_count=args.test_count,
                target_hom=args.target_hom,
                jitter=args.pair_jitter,
                spectral_dim=args.spectral_dim,
                n_max_nodes=args.n_max_nodes,
            )
            save_split(args.output_dir, node_size, split)
            print(f"Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')



if __name__ == "__main__":
    main()
