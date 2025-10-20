"""
Tests for STOAT model implementations.

This module contains unit tests for the STOAT model components.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stoat.models.distributions import GaussianOutput, LaplaceOutput, StudentTOutput, get_distribution_output
from stoat.models.neural_networks import ProbabilisticRNN, ProbabilisticTrainRNN, ProbabilisticPredRNN
from stoat.models.estimators import STOATEstimator, create_stoat_estimator


class TestDistributions(unittest.TestCase):
    """Test distribution outputs."""
    
    def test_gaussian_output(self):
        """Test Gaussian distribution output."""
        distr = GaussianOutput()
        self.assertIsNotNone(distr)
    
    def test_laplace_output(self):
        """Test Laplace distribution output."""
        distr = LaplaceOutput()
        self.assertIsNotNone(distr)
    
    def test_student_t_output(self):
        """Test Student's t distribution output."""
        distr = StudentTOutput()
        self.assertIsNotNone(distr)
    
    def test_get_distribution_output(self):
        """Test distribution factory function."""
        # Test valid distributions
        gaussian = get_distribution_output('gaussian')
        self.assertIsInstance(gaussian, GaussianOutput)
        
        laplace = get_distribution_output('laplace')
        self.assertIsInstance(laplace, LaplaceOutput)
        
        student_t = get_distribution_output('student_t')
        self.assertIsInstance(student_t, StudentTOutput)
        
        # Test invalid distribution
        with self.assertRaises(ValueError):
            get_distribution_output('invalid')


class TestNeuralNetworks(unittest.TestCase):
    """Test neural network implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prediction_length = 10
        self.context_length = 50
        self.num_cells = 32
        self.distr_output = LaplaceOutput()
    
    def test_probabilistic_rnn_init(self):
        """Test ProbabilisticRNN initialization."""
        rnn = ProbabilisticRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells
        )
        
        self.assertEqual(rnn.prediction_length, self.prediction_length)
        self.assertEqual(rnn.context_length, self.context_length)
        self.assertEqual(rnn.num_cells, self.num_cells)
        self.assertEqual(rnn.distr_output, self.distr_output)
    
    def test_probabilistic_train_rnn_init(self):
        """Test ProbabilisticTrainRNN initialization."""
        train_rnn = ProbabilisticTrainRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells
        )
        
        self.assertIsInstance(train_rnn, ProbabilisticRNN)
    
    def test_probabilistic_pred_rnn_init(self):
        """Test ProbabilisticPredRNN initialization."""
        pred_rnn = ProbabilisticPredRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells
        )
        
        self.assertIsInstance(pred_rnn, ProbabilisticTrainRNN)


class TestEstimators(unittest.TestCase):
    """Test estimator implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prediction_length = 10
        self.context_length = 50
        self.freq = "D"
        self.distr_output = LaplaceOutput()
        self.num_cells = 32
    
    def test_stoat_estimator_init(self):
        """Test STOATEstimator initialization."""
        estimator = STOATEstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            distr_output=self.distr_output,
            num_cells=self.num_cells
        )
        
        self.assertEqual(estimator.prediction_length, self.prediction_length)
        self.assertEqual(estimator.context_length, self.context_length)
        self.assertEqual(estimator.freq, self.freq)
        self.assertEqual(estimator.distr_output, self.distr_output)
        self.assertEqual(estimator.num_cells, self.num_cells)
    
    def test_create_stoat_estimator(self):
        """Test factory function for creating STOAT estimator."""
        estimator = create_stoat_estimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distribution="laplace",
            num_cells=self.num_cells,
            epochs=10
        )
        
        self.assertIsInstance(estimator, STOATEstimator)
        self.assertEqual(estimator.prediction_length, self.prediction_length)
        self.assertEqual(estimator.context_length, self.context_length)


class TestIntegration(unittest.TestCase):
    """Integration tests for STOAT components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)
        self.n_regions = 3
        self.n_timepoints = 100
        self.targets = np.random.randn(self.n_regions, self.n_timepoints)
        self.spatial_matrix = np.random.rand(self.n_regions, self.n_regions)
        np.fill_diagonal(self.spatial_matrix, 0)
        
        # Normalize spatial matrix
        row_sums = self.spatial_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.spatial_matrix = self.spatial_matrix / row_sums[:, np.newaxis]
    
    def test_model_initialization(self):
        """Test STOAT model initialization."""
        from stoat import STOAT
        
        model = STOAT(
            prediction_length=10,
            context_length=30,
            spatial_matrix=self.spatial_matrix,
            distribution="laplace",
            num_cells=16,
            epochs=5
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.prediction_length, 10)
        self.assertEqual(model.context_length, 30)
        self.assertTrue(np.array_equal(model.spatial_matrix, self.spatial_matrix))
    
    def test_data_processing(self):
        """Test data processing pipeline."""
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
        self.assertEqual(data['targets'].shape, (3, 50))
        self.assertEqual(data['covariates'].shape, (3, 50, 2))


if __name__ == '__main__':
    unittest.main()
