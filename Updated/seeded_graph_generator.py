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
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
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


def generate_single_graph(
    templates: Sequence[SeedStats],
    n: int,
    target_hom: float,
    jitter: float,
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

        edge_array = np.array(edges, dtype=np.int64)
        edge_index = torch.from_numpy(edge_array.T.copy()).long()
        data = Data(edge_index=edge_index, y=torch.from_numpy(labels).long())
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


def generate_dataset_splits(
    templates: Sequence[SeedStats],
    node_size: int,
    total_graphs: int,
    train_per_set: int,
    test_count: int,
    target_hom: float,
    jitter: float,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
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


def visualize_examples(output_dir: str, node_size: int, n_examples: int = 6, layout: str = 'spring') -> None:
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
            cmap = plt.get_cmap('tab10')
            color_map = {lab: cmap(i % 10) for i, lab in enumerate(uniq)}
            node_colors = [color_map[int(l)] for l in labels]

        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=80, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
        ax.set_title(f"{meta.get('seed','?')} n={n} e={meta.get('num_edges',0)} hom={meta.get('realised_label_hom',0):.2f}")
        ax.set_axis_off()

    out_path = os.path.join(base_dir, f"examples_node_{node_size}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved examples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast seeded synthetic graph dataset generator")
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory for generated splits")
    parser.add_argument("--node-sizes", type=int, nargs="+", default=[100], help="Node counts to generate")
    parser.add_argument("--total-per-size", type=int, default=5100, help="Graphs to sample per node size")
    parser.add_argument("--train-per-set", type=int, default=2500, help="Graphs per training subset")
    parser.add_argument("--test-count", type=int, default=100, help="Graphs reserved for evaluation")
    parser.add_argument("--target-hom", type=float, default=0.5, help="Desired label homophily")
    parser.add_argument("--pair-jitter", type=float, default=0.1, help="Dirichlet noise level for pair probabilities")
    parser.add_argument("--planetoid-root", type=str, default="./planetoid", help="Planetoid dataset root directory")
    parser.add_argument("--visualize", action="store_true", help="Generate example visualizations after creating splits")
    parser.add_argument("--viz-examples", type=int, default=6, help="Number of example graphs to visualize (default: 6)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)

    templates = [
        load_seed_stats("Cora", args.planetoid_root),
        load_seed_stats("CiteSeer", args.planetoid_root),
    ]

    for node_size in args.node_sizes:
        print(f"Generating node size {node_size}...")
        split = generate_dataset_splits(
            templates=templates,
            node_size=node_size,
            total_graphs=args.total_per_size,
            train_per_set=args.train_per_set,
            test_count=args.test_count,
            target_hom=args.target_hom,
            jitter=args.pair_jitter,
        )
        save_split(args.output_dir, node_size, split)
        print(f"Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
        
        # Auto-visualize if requested
        if args.visualize:
            print(f"Creating visualizations for node size {node_size}...")
            visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')



if __name__ == "__main__":
    main()
