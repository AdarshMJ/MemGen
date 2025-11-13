"""
Preprocessing script for custom node_500 dataset.
Converts PyG Data objects to NetworkX graphs for GruM training.
"""
import torch
from torch.utils.data import random_split
import networkx as nx
import pickle
import os
import sys
from torch_geometric.utils import to_networkx

def load_node500_data(split_file='GruM_2D/data/node_500/S1.pt'):
    """Load the node_500 dataset from .pt file."""
    print(f'Loading data from {split_file}...')
    # Load with weights_only=False since it contains PyG Data objects
    data_list = torch.load(split_file, weights_only=False)
    print(f'Loaded {len(data_list)} graphs')
    
    # Extract PyG Data objects (first element of each tuple)
    pyg_graphs = [item[0] for item in data_list]
    
    return pyg_graphs


def pyg_to_networkx(pyg_graphs):
    """Convert PyG Data objects to NetworkX graphs."""
    graph_list = []
    for data in pyg_graphs:
        # Convert PyG data to NetworkX graph
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        # Remove isolated nodes if any
        G.remove_nodes_from(list(nx.isolates(G)))
        graph_list.append(G)
    
    return graph_list


def preprocess_node500(data_dir='GruM_2D/data', split='S1', test_ratio=0.2, val_ratio=0.1):
    """
    Preprocess the node_500 dataset for GruM training.
    
    Args:
        data_dir: Directory containing the data
        split: Which split to use (S1 or S2)
        test_ratio: Ratio of test set
        val_ratio: Ratio of validation set (from remaining after test split)
    """
    split_file = f'{data_dir}/node_500/{split}.pt'
    
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f'Dataset file {split_file} not found.')
    
    # Load PyG graphs
    pyg_graphs = load_node500_data(split_file)
    
    # Convert to NetworkX
    print('Converting PyG graphs to NetworkX format...')
    graph_list = pyg_to_networkx(pyg_graphs)
    print(f'Converted {len(graph_list)} graphs')
    
    # Print statistics
    num_nodes = [G.number_of_nodes() for G in graph_list]
    num_edges = [G.number_of_edges() for G in graph_list]
    print(f'\nDataset Statistics:')
    print(f'Number of graphs: {len(graph_list)}')
    print(f'Avg nodes: {sum(num_nodes)/len(num_nodes):.2f}')
    print(f'Min/Max nodes: {min(num_nodes)}/{max(num_nodes)}')
    print(f'Avg edges: {sum(num_edges)/len(num_edges):.2f}')
    print(f'Min/Max edges: {min(num_edges)}/{max(num_edges)}')
    
    # Split into train/val/test
    total_len = len(graph_list)
    test_len = int(total_len * test_ratio)
    remaining_len = total_len - test_len
    val_len = int(remaining_len * val_ratio)
    train_len = remaining_len - val_len
    
    print(f'\nSplitting dataset:')
    print(f'Train: {train_len}, Val: {val_len}, Test: {test_len}')
    
    # Use indices to maintain order
    indices = list(range(total_len))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]
    
    train_graphs = [graph_list[i] for i in train_indices]
    val_graphs = [graph_list[i] for i in val_indices]
    test_graphs = [graph_list[i] for i in test_indices]
    
    # Save as pickle file
    output_file = f'{data_dir}/node500.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(obj=(train_graphs, val_graphs, test_graphs), 
                   file=f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'\nSaved preprocessed data to {output_file}')
    print(f'Dataset is ready for training!')
    
    return train_graphs, val_graphs, test_graphs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess node_500 dataset for GruM')
    parser.add_argument('--split', type=str, default='S1', 
                       choices=['S1', 'S2'],
                       help='Which split to use (S1 or S2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of test set')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation set from remaining data')
    args = parser.parse_args()
    
    preprocess_node500(split=args.split, 
                      test_ratio=args.test_ratio,
                      val_ratio=args.val_ratio)
