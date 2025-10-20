"""
Tests for STOAT utility functions.

This module contains unit tests for the STOAT utility functions.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stoat.utils.spatial import (
    create_spatial_matrix, 
    compute_spatial_weights, 
    validate_spatial_matrix,
    get_spatial_statistics
)
from stoat.utils.evaluation import STOATEvaluator


class TestSpatialUtils(unittest.TestCase):
    """Test spatial utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_regions = 5
        self.coordinates = np.random.rand(self.n_regions, 2) * 100
        self.regions = [f"Region_{i}" for i in range(self.n_regions)]
    
    def test_create_spatial_matrix_distance(self):
        """Test creating spatial matrix using distance method."""
        S = create_spatial_matrix(
            regions=self.regions,
            method="distance",
            coordinates=self.coordinates,
            threshold=50.0
        )
        
        self.assertEqual(S.shape, (self.n_regions, self.n_regions))
        self.assertTrue(np.allclose(np.diag(S), 0))  # Diagonal should be zero
        self.assertTrue(np.all(S >= 0))  # All weights should be non-negative
    
    def test_create_spatial_matrix_binary(self):
        """Test creating spatial matrix using binary method."""
        S = create_spatial_matrix(
            regions=self.regions,
            method="binary",
            coordinates=self.coordinates,
            threshold=50.0
        )
        
        self.assertEqual(S.shape, (self.n_regions, self.n_regions))
        self.assertTrue(np.allclose(np.diag(S), 0))
        self.assertTrue(np.all((S == 0) | (S == 1)))  # Binary values only
    
    def test_create_spatial_matrix_knn(self):
        """Test creating spatial matrix using k-NN method."""
        S = create_spatial_matrix(
            regions=self.regions,
            method="knn",
            coordinates=self.coordinates,
            k_nearest=2
        )
        
        self.assertEqual(S.shape, (self.n_regions, self.n_regions))
        self.assertTrue(np.allclose(np.diag(S), 0))
        
        # Each row should have exactly k non-zero entries
        for i in range(self.n_regions):
            self.assertEqual(np.sum(S[i, :] > 0), 2)
    
    def test_create_spatial_matrix_custom(self):
        """Test creating spatial matrix using custom weights."""
        custom_weights = np.random.rand(self.n_regions, self.n_regions)
        np.fill_diagonal(custom_weights, 0)
        
        S = create_spatial_matrix(
            regions=self.regions,
            method="custom",
            weights=custom_weights
        )
        
        self.assertTrue(np.array_equal(S, custom_weights))
    
    def test_compute_spatial_weights(self):
        """Test computing spatial weights."""
        weights = compute_spatial_weights(
            coordinates=self.coordinates,
            method="gaussian",
            bandwidth=10.0
        )
        
        self.assertEqual(weights.shape, (self.n_regions, self.n_regions))
        self.assertTrue(np.allclose(np.diag(weights), 0))
        self.assertTrue(np.all(weights >= 0))
    
    def test_validate_spatial_matrix(self):
        """Test spatial matrix validation."""
        # Valid matrix
        valid_matrix = np.random.rand(self.n_regions, self.n_regions)
        np.fill_diagonal(valid_matrix, 0)
        valid_matrix = np.abs(valid_matrix)
        
        self.assertTrue(validate_spatial_matrix(valid_matrix))
        
        # Invalid matrix (not square)
        invalid_matrix = np.random.rand(self.n_regions, self.n_regions + 1)
        self.assertFalse(validate_spatial_matrix(invalid_matrix))
        
        # Invalid matrix (negative values)
        invalid_matrix = np.random.randn(self.n_regions, self.n_regions)
        np.fill_diagonal(invalid_matrix, 0)
        self.assertFalse(validate_spatial_matrix(invalid_matrix))
    
    def test_get_spatial_statistics(self):
        """Test getting spatial matrix statistics."""
        S = create_spatial_matrix(
            regions=self.regions,
            method="distance",
            coordinates=self.coordinates,
            threshold=50.0
        )
        
        stats = get_spatial_statistics(S)
        
        self.assertIn('shape', stats)
        self.assertIn('density', stats)
        self.assertIn('mean_weight', stats)
        self.assertIn('max_weight', stats)
        self.assertIn('row_sums', stats)
        self.assertIn('col_sums', stats)
        self.assertIn('is_symmetric', stats)
        self.assertIn('is_normalized', stats)
        
        self.assertEqual(stats['shape'], (self.n_regions, self.n_regions))


class TestEvaluationUtils(unittest.TestCase):
    """Test evaluation utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = STOATEvaluator()
        self.n_regions = 3
        self.prediction_length = 10
        
        # Create mock forecasts
        self.forecasts = []
        self.targets = np.random.randn(self.n_regions, self.prediction_length)
        
        for i in range(self.n_regions):
            forecast = Mock()
            forecast.mean = np.random.randn(self.prediction_length)
            forecast.quantile = Mock(return_value=np.random.randn(self.prediction_length))
            forecast.samples = np.random.randn(100, self.prediction_length)
            forecast.num_samples = 100
            forecast.prediction_length = self.prediction_length
            self.forecasts.append(forecast)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = STOATEvaluator(quantiles=[0.1, 0.5, 0.9])
        self.assertEqual(evaluator.quantiles, [0.1, 0.5, 0.9])
    
    def test_compute_forecast_metrics(self):
        """Test computing forecast metrics."""
        metrics = self.evaluator.compute_forecast_metrics(
            self.forecasts, 
            self.targets
        )
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('mean_std', metrics)
        
        # Check that metrics are non-negative
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.assertGreaterEqual(value, 0)
    
    def test_evaluate_causal_inference(self):
        """Test causal inference evaluation."""
        causal_params = {
            'delta': -0.15,
            'rho': 0.4,
            'gamma': np.array([0.5, -0.3, 0.2])
        }
        
        evaluation = self.evaluator.evaluate_causal_inference(causal_params)
        
        self.assertIn('treatment_significance', evaluation)
        self.assertIn('spatial_significance', evaluation)
    
    def test_compute_spatial_metrics(self):
        """Test computing spatial metrics."""
        spatial_matrix = np.random.rand(self.n_regions, self.n_regions)
        np.fill_diagonal(spatial_matrix, 0)
        
        metrics = self.evaluator.compute_spatial_metrics(
            self.forecasts,
            self.targets,
            spatial_matrix
        )
        
        self.assertIn('spatial_error_correlation', metrics)
        self.assertIn('forecast_spatial_autocorr', metrics)
        self.assertIn('true_spatial_autocorr', metrics)
    
    def test_evaluate_model_interpretability(self):
        """Test model interpretability evaluation."""
        causal_params = {'delta': -0.2, 'rho': 0.5}
        forecast_params = {'num_cells': 64}
        
        interpretability = self.evaluator.evaluate_model_interpretability(
            causal_params, 
            forecast_params
        )
        
        self.assertIn('treatment_interpretation', interpretability)
        self.assertIn('spatial_interpretation', interpretability)
        self.assertIn('model_complexity', interpretability)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def test_create_synthetic_data(self):
        """Test creating synthetic data."""
        from stoat.data import create_synthetic_data
        
        data = create_synthetic_data(
            n_regions=3,
            n_timepoints=50,
            n_covariates=2,
            random_state=42
        )
        
        self.assertIn('targets', data)
        self.assertIn('covariates', data)
        self.assertIn('regions', data)
        self.assertIn('dates', data)
        
        self.assertEqual(data['targets'].shape, (3, 50))
        self.assertEqual(data['covariates'].shape, (3, 50, 2))
        self.assertEqual(len(data['regions']), 3)
        self.assertEqual(len(data['dates']), 50)
    
    def test_prepare_dataset(self):
        """Test preparing dataset in GluonTS format."""
        from stoat.data import prepare_dataset
        
        targets = np.random.randn(3, 50)
        covariates = np.random.randn(3, 50, 2)
        
        dataset = prepare_dataset(
            targets=targets,
            covariates=covariates,
            freq="D"
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(len(list(dataset)), 3)


if __name__ == '__main__':
    unittest.main()
