"""
Visualization utilities for STOAT.

This module provides visualization functions for STOAT results,
including forecast plots and causal effect visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import warnings


def plot_forecasts(
    forecasts: List,
    targets: np.ndarray,
    regions: Optional[List[str]] = None,
    plot_length: int = 50,
    prediction_intervals: Tuple[float, float] = (50.0, 90.0),
    figsize: Tuple[int, int] = (15, 10),
    title: str = "STOAT Forecasts"
) -> None:
    """
    Plot forecasts with prediction intervals.
    
    Args:
        forecasts: List of forecast objects
        targets: True target values (N x T)
        regions: List of region names
        plot_length: Length of historical data to plot
        prediction_intervals: Prediction intervals to show
        figsize: Figure size
        title: Plot title
    """
    N = len(forecasts)
    n_cols = min(3, N)
    n_rows = (N + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if N == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (forecast, target) in enumerate(zip(forecasts, targets)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot historical data
        hist_length = min(plot_length, len(target))
        ax.plot(range(hist_length), target[-hist_length:], 'b-', label='Historical', linewidth=2)
        
        # Plot forecast
        forecast.plot(prediction_intervals=prediction_intervals, color='r', ax=ax)
        
        # Add true values for comparison if available
        if len(target) > hist_length:
            true_future = target[hist_length:]
            ax.plot(range(hist_length, hist_length + len(true_future)), 
                   true_future, 'g--', label='True', linewidth=2)
        
        # Set title
        if regions and i < len(regions):
            ax.set_title(f'{regions[i]}')
        else:
            ax.set_title(f'Series {i+1}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(N, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_causal_effects(
    causal_parameters: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot causal effect parameters.
    
    Args:
        causal_parameters: Dictionary with causal parameters
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Treatment effect
    if 'delta' in causal_parameters:
        ax = axes[0]
        delta = causal_parameters['delta']
        ax.bar(['Treatment Effect'], [delta], color='red' if delta < 0 else 'blue')
        ax.set_title('Treatment Effect (δ)')
        ax.set_ylabel('Effect Size')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    # Spatial coefficient
    if 'rho' in causal_parameters:
        ax = axes[1]
        rho = causal_parameters['rho']
        ax.bar(['Spatial Coefficient'], [rho], color='green')
        ax.set_title('Spatial Coefficient (ρ)')
        ax.set_ylabel('Coefficient Value')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    # Covariate effects
    if 'gamma' in causal_parameters:
        ax = axes[2]
        gamma = causal_parameters['gamma']
        if isinstance(gamma, np.ndarray):
            covariate_names = [f'Covariate {i+1}' for i in range(len(gamma))]
            colors = ['red' if g < 0 else 'blue' for g in gamma]
            ax.bar(covariate_names, gamma, color=colors)
            ax.set_title('Covariate Effects (γ)')
            ax.set_ylabel('Effect Size')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    # Parameter significance
    ax = axes[3]
    significance_data = []
    significance_labels = []
    
    if 'delta' in causal_parameters:
        delta = causal_parameters['delta']
        significance_data.append(abs(delta))
        significance_labels.append('Treatment\nEffect')
    
    if 'rho' in causal_parameters:
        rho = causal_parameters['rho']
        significance_data.append(abs(rho))
        significance_labels.append('Spatial\nCoefficient')
    
    if significance_data:
        ax.bar(significance_labels, significance_data, color='orange')
        ax.set_title('Parameter Significance')
        ax.set_ylabel('Absolute Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_spatial_network(
    spatial_matrix: np.ndarray,
    regions: Optional[List[str]] = None,
    coordinates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Spatial Network"
) -> None:
    """
    Plot spatial network structure.
    
    Args:
        spatial_matrix: Spatial relation matrix (N x N)
        regions: List of region names
        coordinates: Coordinate matrix (N x 2) for positioning
        figsize: Figure size
        title: Plot title
    """
    N = spatial_matrix.shape[0]
    
    if coordinates is None:
        # Generate circular layout
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        coordinates = np.column_stack([np.cos(angles), np.sin(angles)])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot edges
    for i in range(N):
        for j in range(N):
            if spatial_matrix[i, j] > 0 and i != j:
                weight = spatial_matrix[i, j]
                ax.plot([coordinates[i, 0], coordinates[j, 0]], 
                       [coordinates[i, 1], coordinates[j, 1]], 
                       'b-', alpha=min(weight * 2, 1.0), linewidth=weight * 3)
    
    # Plot nodes
    ax.scatter(coordinates[:, 0], coordinates[:, 1], 
              c='red', s=100, alpha=0.7, zorder=5)
    
    # Add region labels
    if regions:
        for i, region in enumerate(regions):
            ax.annotate(region, (coordinates[i, 0], coordinates[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(
    training_history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot training history.
    
    Args:
        training_history: Dictionary with training metrics
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if 'loss' in training_history:
        ax = axes[0]
        ax.plot(training_history['loss'], 'b-', label='Training Loss')
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot other metrics
    if len(training_history) > 1:
        ax = axes[1]
        for key, values in training_history.items():
            if key != 'loss':
                ax.plot(values, label=key)
        ax.set_title('Training Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics(
    metrics: Dict[str, float],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        figsize: Figure size
    """
    # Separate metrics by type
    forecast_metrics = {}
    causal_metrics = {}
    spatial_metrics = {}
    
    for key, value in metrics.items():
        if any(x in key.lower() for x in ['mse', 'mae', 'rmse', 'mape', 'smape']):
            forecast_metrics[key] = value
        elif any(x in key.lower() for x in ['delta', 'rho', 'gamma', 'treatment', 'spatial']):
            causal_metrics[key] = value
        else:
            spatial_metrics[key] = value
    
    n_plots = sum([len(forecast_metrics) > 0, len(causal_metrics) > 0, len(spatial_metrics) > 0])
    
    if n_plots == 0:
        warnings.warn("No metrics to plot")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot forecast metrics
    if forecast_metrics:
        ax = axes[plot_idx]
        keys = list(forecast_metrics.keys())
        values = list(forecast_metrics.values())
        ax.bar(keys, values, color='blue', alpha=0.7)
        ax.set_title('Forecast Metrics')
        ax.set_ylabel('Metric Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot causal metrics
    if causal_metrics:
        ax = axes[plot_idx]
        keys = list(causal_metrics.keys())
        values = list(causal_metrics.values())
        colors = ['red' if v < 0 else 'blue' for v in values]
        ax.bar(keys, values, color=colors, alpha=0.7)
        ax.set_title('Causal Metrics')
        ax.set_ylabel('Metric Value')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot spatial metrics
    if spatial_metrics:
        ax = axes[plot_idx]
        keys = list(spatial_metrics.keys())
        values = list(spatial_metrics.values())
        ax.bar(keys, values, color='green', alpha=0.7)
        ax.set_title('Spatial Metrics')
        ax.set_ylabel('Metric Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
