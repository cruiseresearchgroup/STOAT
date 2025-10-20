"""
STOAT utility modules.

This package contains utility functions for STOAT, including
spatial operations, evaluation metrics, and visualization tools.
"""

from .spatial import (
    create_spatial_matrix,
    compute_spatial_weights,
    validate_spatial_matrix,
    get_spatial_statistics,
    plot_spatial_matrix,
    create_contiguity_matrix
)
from .evaluation import STOATEvaluator

# Import visualization functions if available
try:
    from .visualization import plot_forecasts, plot_causal_effects
    __all__ = [
        'create_spatial_matrix',
        'compute_spatial_weights',
        'validate_spatial_matrix',
        'get_spatial_statistics',
        'plot_spatial_matrix',
        'create_contiguity_matrix',
        'STOATEvaluator',
        'plot_forecasts',
        'plot_causal_effects'
    ]
except ImportError:
    __all__ = [
        'create_spatial_matrix',
        'compute_spatial_weights',
        'validate_spatial_matrix',
        'get_spatial_statistics',
        'plot_spatial_matrix',
        'create_contiguity_matrix',
        'STOATEvaluator'
    ]
