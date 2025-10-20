"""
Spatial utility functions for STOAT.

This module provides utilities for creating and manipulating spatial relation matrices
and computing spatial weights for the STOAT framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import warnings


def create_spatial_matrix(
    regions: Union[List, np.ndarray],
    method: str = "distance",
    threshold: Optional[float] = None,
    k_nearest: Optional[int] = None,
    coordinates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Create spatial relation matrix for STOAT.
    
    Args:
        regions: List or array of region identifiers
        method: Method for creating spatial weights ('distance', 'knn', 'custom', 'binary')
        threshold: Distance threshold for binary weights
        k_nearest: Number of nearest neighbors for k-NN weights
        coordinates: Coordinate matrix (N x 2) for distance-based methods
        weights: Custom weight matrix (N x N)
        normalize: Whether to normalize the spatial matrix
        **kwargs: Additional arguments
        
    Returns:
        Spatial relation matrix (N x N)
    """
    N = len(regions)
    
    if method == "custom":
        if weights is None:
            raise ValueError("Custom weights must be provided for 'custom' method")
        S = np.array(weights)
    elif method == "binary":
        if coordinates is None:
            raise ValueError("Coordinates must be provided for 'binary' method")
        S = _create_binary_weights(coordinates, threshold)
    elif method == "distance":
        if coordinates is None:
            raise ValueError("Coordinates must be provided for 'distance' method")
        S = _create_distance_weights(coordinates, threshold, **kwargs)
    elif method == "knn":
        if coordinates is None:
            raise ValueError("Coordinates must be provided for 'knn' method")
        S = _create_knn_weights(coordinates, k_nearest)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Ensure matrix is square
    if S.shape != (N, N):
        raise ValueError(f"Spatial matrix must be {N}x{N}, got {S.shape}")
    
    # Normalize if requested
    if normalize:
        S = _normalize_spatial_matrix(S)
    
    return S


def _create_binary_weights(
    coordinates: np.ndarray, 
    threshold: float
) -> np.ndarray:
    """
    Create binary spatial weights based on distance threshold.
    
    Args:
        coordinates: Coordinate matrix (N x 2)
        threshold: Distance threshold
        
    Returns:
        Binary spatial weight matrix
    """
    distances = pdist(coordinates)
    distance_matrix = squareform(distances)
    
    # Create binary weights
    weights = (distance_matrix <= threshold).astype(float)
    
    # Remove self-connections
    np.fill_diagonal(weights, 0)
    
    return weights


def _create_distance_weights(
    coordinates: np.ndarray,
    threshold: Optional[float] = None,
    power: float = 1.0,
    inverse: bool = True
) -> np.ndarray:
    """
    Create distance-based spatial weights.
    
    Args:
        coordinates: Coordinate matrix (N x 2)
        threshold: Distance threshold (optional)
        power: Power for distance transformation
        inverse: Whether to use inverse distance weights
        
    Returns:
        Distance-based spatial weight matrix
    """
    distances = pdist(coordinates)
    distance_matrix = squareform(distances)
    
    # Apply threshold if provided
    if threshold is not None:
        distance_matrix = np.where(distance_matrix <= threshold, distance_matrix, np.inf)
    
    # Transform distances
    if inverse:
        weights = 1.0 / (distance_matrix ** power)
        weights[np.isinf(weights)] = 0
    else:
        weights = distance_matrix ** power
        weights[np.isinf(weights)] = 0
    
    # Remove self-connections
    np.fill_diagonal(weights, 0)
    
    return weights


def _create_knn_weights(
    coordinates: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Create k-nearest neighbor spatial weights.
    
    Args:
        coordinates: Coordinate matrix (N x 2)
        k: Number of nearest neighbors
        
    Returns:
        k-NN spatial weight matrix
    """
    distances = pdist(coordinates)
    distance_matrix = squareform(distances)
    
    # Find k nearest neighbors for each region
    weights = np.zeros_like(distance_matrix)
    
    for i in range(len(coordinates)):
        # Get indices of k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distance_matrix[i])[1:k+1]
        weights[i, nearest_indices] = 1.0
    
    return weights


def _normalize_spatial_matrix(S: np.ndarray) -> np.ndarray:
    """
    Normalize spatial matrix by row sums.
    
    Args:
        S: Spatial matrix
        
    Returns:
        Normalized spatial matrix
    """
    row_sums = S.sum(axis=1)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    return S / row_sums[:, np.newaxis]


def compute_spatial_weights(
    coordinates: np.ndarray,
    method: str = "gaussian",
    bandwidth: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Compute spatial weights using various methods.
    
    Args:
        coordinates: Coordinate matrix (N x 2)
        method: Weight computation method ('gaussian', 'exponential', 'linear')
        bandwidth: Bandwidth parameter
        **kwargs: Additional arguments
        
    Returns:
        Spatial weight matrix
    """
    distances = pdist(coordinates)
    distance_matrix = squareform(distances)
    
    if method == "gaussian":
        if bandwidth is None:
            bandwidth = np.median(distances)
        weights = np.exp(-(distance_matrix ** 2) / (2 * bandwidth ** 2))
    elif method == "exponential":
        if bandwidth is None:
            bandwidth = np.median(distances)
        weights = np.exp(-distance_matrix / bandwidth)
    elif method == "linear":
        if bandwidth is None:
            bandwidth = np.max(distances)
        weights = np.maximum(0, 1 - distance_matrix / bandwidth)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Remove self-connections
    np.fill_diagonal(weights, 0)
    
    return weights


def validate_spatial_matrix(S: np.ndarray) -> bool:
    """
    Validate spatial matrix properties.
    
    Args:
        S: Spatial matrix to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if square
    if S.shape[0] != S.shape[1]:
        warnings.warn("Spatial matrix is not square")
        return False
    
    # Check if non-negative
    if np.any(S < 0):
        warnings.warn("Spatial matrix contains negative values")
        return False
    
    # Check if diagonal is zero
    if not np.allclose(np.diag(S), 0):
        warnings.warn("Spatial matrix diagonal is not zero")
        return False
    
    # Check if symmetric (optional)
    if not np.allclose(S, S.T):
        warnings.warn("Spatial matrix is not symmetric")
    
    return True


def get_spatial_statistics(S: np.ndarray) -> Dict:
    """
    Get statistics about the spatial matrix.
    
    Args:
        S: Spatial matrix
        
    Returns:
        Dictionary with spatial statistics
    """
    stats = {
        'shape': S.shape,
        'density': np.count_nonzero(S) / (S.shape[0] * S.shape[1]),
        'mean_weight': np.mean(S[S > 0]) if np.any(S > 0) else 0,
        'max_weight': np.max(S),
        'min_weight': np.min(S[S > 0]) if np.any(S > 0) else 0,
        'row_sums': S.sum(axis=1),
        'col_sums': S.sum(axis=0),
        'is_symmetric': np.allclose(S, S.T),
        'is_normalized': np.allclose(S.sum(axis=1), 1.0)
    }
    
    return stats


def plot_spatial_matrix(
    S: np.ndarray,
    regions: Optional[List] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Spatial Relation Matrix"
) -> None:
    """
    Plot spatial relation matrix.
    
    Args:
        S: Spatial matrix
        regions: List of region names
        figsize: Figure size
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(S, cmap='Blues', aspect='auto')
    
    if regions is not None:
        ax.set_xticks(range(len(regions)))
        ax.set_yticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.set_yticklabels(regions)
    
    ax.set_xlabel('Region')
    ax.set_ylabel('Region')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spatial Weight')
    
    plt.tight_layout()
    plt.show()


def create_contiguity_matrix(
    regions: List,
    adjacency_data: Optional[Dict] = None
) -> np.ndarray:
    """
    Create contiguity-based spatial matrix.
    
    Args:
        regions: List of region identifiers
        adjacency_data: Dictionary mapping regions to their neighbors
        
    Returns:
        Contiguity matrix
    """
    N = len(regions)
    S = np.zeros((N, N))
    
    if adjacency_data is None:
        warnings.warn("No adjacency data provided. Returning zero matrix.")
        return S
    
    # Create region index mapping
    region_to_idx = {region: i for i, region in enumerate(regions)}
    
    # Fill adjacency matrix
    for region, neighbors in adjacency_data.items():
        if region in region_to_idx:
            i = region_to_idx[region]
            for neighbor in neighbors:
                if neighbor in region_to_idx:
                    j = region_to_idx[neighbor]
                    S[i, j] = 1.0
    
    return S
