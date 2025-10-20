"""
STOAT: Spatial-Temporal Causal Inference for Epidemic Forecasting

A framework that combines spatial causal inference with deep probabilistic forecasting
for epidemic prediction, incorporating spatial dependencies and uncertainty quantification.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.stoat import STOAT
from .core.spatial_causal import SpatialCausalInference
from .core.forecasting import DeepProbabilisticForecasting

# Model imports
from .models.distributions import (
    GaussianOutput,
    LaplaceOutput,
    StudentTOutput,
)
from .models.neural_networks import (
    ProbabilisticRNN,
    ProbabilisticTrainRNN,
    ProbabilisticPredRNN,
)
from .models.estimators import STOATEstimator

# Data processing imports
from .data.processors import EpidemicDataProcessor
from .data.loaders import load_epidemic_data

# Utility imports
from .utils.spatial import create_spatial_matrix, compute_spatial_weights
from .utils.evaluation import STOATEvaluator
from .utils.visualization import plot_forecasts, plot_causal_effects

__all__ = [
    # Core
    "STOAT",
    "SpatialCausalInference", 
    "DeepProbabilisticForecasting",
    
    # Models
    "GaussianOutput",
    "LaplaceOutput", 
    "StudentTOutput",
    "ProbabilisticRNN",
    "ProbabilisticTrainRNN",
    "ProbabilisticPredRNN",
    "STOATEstimator",
    
    # Data
    "EpidemicDataProcessor",
    "load_epidemic_data",
    
    # Utils
    "create_spatial_matrix",
    "compute_spatial_weights",
    "STOATEvaluator",
    "plot_forecasts",
    "plot_causal_effects",
]
