"""
Dataset loader for multi-graph training
Loads graphs from data/node_xx folders
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import os


class GraphDataset(Dataset):
    """
    Dataset that loads multiple graphs from .pt files
    
    Args:
        data_path: Path to .pt file (e.g., 'data/node_20/S1.pt')
    """
    def __init__(self, data_path):
        self.data_path = data_path
        print(f"Loading dataset from {data_path}...")
        self.graphs = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.graphs)} graphs")
        
        # Extract first graph to get dimensions
        sample_graph, sample_meta = self.graphs[0]
        self.num_node_features = sample_graph.x.shape[1]
        self.num_classes = len(torch.unique(sample_graph.y))
        self.num_nodes = sample_graph.num_nodes
        
        print(f"Graph properties: {self.num_nodes} nodes, "
              f"{self.num_node_features} features, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """
        Returns:
            graph: PyG Data object with x, edge_index, y
            metadata: dict with stats
        """
        graph, metadata = self.graphs[idx]
        return graph, metadata


def collate_graphs(batch):
    """
    Custom collate function for batching graphs
    
    Args:
        batch: List of (graph, metadata) tuples
    
    Returns:
        batched_graphs: PyG Batch object
        metadata_list: List of metadata dicts
    """
    graphs = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    
    # Use PyG's Batch to create a disconnected union of graphs
    batched_graphs = Batch.from_data_list(graphs)
    
    return batched_graphs, metadata


def get_edge_marginals(dataset):
    """
    Compute marginal distribution of edge existence across all graphs
    
    Returns:
        E_marginal: [2] tensor with [P(no edge), P(edge)]
    """
    total_possible_edges = 0
    total_existing_edges = 0
    
    for graph, _ in dataset:
        n = graph.num_nodes
        total_possible_edges += n * (n - 1) // 2  # Undirected
        total_existing_edges += graph.num_edges
    
    p_edge = total_existing_edges / total_possible_edges
    p_no_edge = 1 - p_edge
    
    E_marginal = torch.tensor([p_no_edge, p_edge])
    print(f"Edge marginals: P(no edge)={p_no_edge:.4f}, P(edge)={p_edge:.4f}")
    
    return E_marginal


def get_label_marginals(dataset):
    """
    Compute marginal distribution of node labels
    
    Returns:
        Y_marginal: [num_classes] tensor
    """
    label_counts = {}
    total_nodes = 0
    
    for graph, _ in dataset:
        for label in graph.y.tolist():
            label_counts[label] = label_counts.get(label, 0) + 1
            total_nodes += 1
    
    num_classes = max(label_counts.keys()) + 1
    Y_marginal = torch.zeros(num_classes)
    
    for label, count in label_counts.items():
        Y_marginal[label] = count / total_nodes
    
    print(f"Label marginals: {Y_marginal}")
    
    return Y_marginal


if __name__ == "__main__":
    # Test the dataset loader
    dataset = GraphDataset('data/node_20/S1.pt')
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting a sample
    graph, meta = dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Features shape: {graph.x.shape}")
    print(f"  Labels shape: {graph.y.shape}")
    print(f"  Metadata: {meta.keys()}")
    
    # Test marginals
    E_marginal = get_edge_marginals(dataset)
    Y_marginal = get_label_marginals(dataset)
    
    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_graphs)
    
    batched_graphs, metadata = next(iter(loader))
    print(f"\nBatched graphs:")
    print(f"  Total nodes in batch: {batched_graphs.num_nodes}")
    print(f"  Total edges in batch: {batched_graphs.num_edges}")
    print(f"  Batch vector: {batched_graphs.batch}")
    print(f"  Number of graphs: {batched_graphs.num_graphs}")
