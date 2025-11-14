"""
__init__.py for lightweight_graphmaker package
"""

from .dataset import GraphDataset, get_edge_marginals, get_label_marginals
from .diffusion import GraphDiffusion, NoiseSchedule, MarginalTransition
from .model import BiasFreeDenoisingGNN
from .train import DiffusionTrainer, train_model
from .sample import GraphSampler, generate_from_checkpoint
from .wl_kernel import (
    compute_wl_similarity,
    evaluate_memorization_vs_generalization,
    pyg_to_networkx,
    find_closest_matches
)
from .visualize import (
    visualize_graph,
    visualize_graph_pairs,
    plot_wl_similarity_histogram,
    create_comprehensive_visualization
)

__version__ = "1.0.0"

__all__ = [
    # Dataset
    'GraphDataset',
    'get_edge_marginals',
    'get_label_marginals',
    
    # Diffusion
    'GraphDiffusion',
    'NoiseSchedule',
    'MarginalTransition',
    
    # Model
    'BiasFreeDenoisingGNN',
    
    # Training
    'DiffusionTrainer',
    'train_model',
    
    # Sampling
    'GraphSampler',
    'generate_from_checkpoint',
    
    # Evaluation
    'compute_wl_similarity',
    'evaluate_memorization_vs_generalization',
    'pyg_to_networkx',
    'find_closest_matches',
    
    # Visualization
    'visualize_graph',
    'visualize_graph_pairs',
    'plot_wl_similarity_histogram',
    'create_comprehensive_visualization',
]
