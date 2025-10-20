"""
Deep Probabilistic Forecasting Module for STOAT.

This module implements the deep probabilistic forecasting component that leverages
causally adjusted spatial-temporal representations for epidemic forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.predictor import Predictor
import warnings

from ..models.estimators import STOATEstimator, create_stoat_estimator
from ..models.distributions import get_distribution_output


class DeepProbabilisticForecasting:
    """
    Deep Probabilistic Forecasting module for STOAT.
    
    This class implements the neural encoder-decoder architecture that processes
    causally adjusted spatial-temporal representations for probabilistic forecasting.
    """
    
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        freq: str = "D",
        distribution: str = "laplace",
        num_cells: int = 64,
        num_sample_paths: int = 100,
        scaling: bool = True,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        ctx: str = "cpu",
        **kwargs
    ):
        """
        Initialize the deep probabilistic forecasting module.
        
        Args:
            prediction_length: Length of prediction horizon
            context_length: Length of context window
            freq: Frequency of the time series
            distribution: Distribution type ('gaussian', 'laplace', 'student_t')
            num_cells: Number of LSTM cells
            num_sample_paths: Number of sample paths for prediction
            scaling: Whether to use scaling
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            ctx: Context (device) for training
            **kwargs: Additional arguments
        """
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distribution = distribution
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ctx = ctx
        
        # Initialize estimator
        self.estimator = None
        self.predictor = None
        
        # Training history
        self.training_history = None
        
    def _prepare_dataset(
        self, 
        targets: np.ndarray, 
        covariates: Optional[np.ndarray] = None,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None
    ) -> ListDataset:
        """
        Prepare dataset in GluonTS format.
        
        Args:
            targets: Target time series (N x T)
            covariates: Covariate time series (N x T x K)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            
        Returns:
            GluonTS ListDataset
        """
        N, T = targets.shape
        
        if start_dates is None:
            start_dates = [pd.Timestamp('2020-01-01', freq=self.freq) for _ in range(N)]
        
        if feat_static_cat is None:
            feat_static_cat = np.arange(N)
        
        # Prepare dataset entries
        dataset_entries = []
        for i in range(N):
            entry = {
                FieldName.TARGET: targets[i, :],
                FieldName.START: start_dates[i],
                FieldName.FEAT_STATIC_CAT: [feat_static_cat[i]]
            }
            
            # Add dynamic real features (covariates)
            if covariates is not None:
                if covariates.ndim == 3:  # (N, T, K)
                    entry[FieldName.FEAT_DYNAMIC_REAL] = covariates[i, :, :].T
                elif covariates.ndim == 2:  # (N*T, K) - need to reshape
                    cov_reshaped = covariates.reshape(N, T, -1)
                    entry[FieldName.FEAT_DYNAMIC_REAL] = cov_reshaped[i, :, :].T
            
            dataset_entries.append(entry)
        
        return ListDataset(dataset_entries, freq=self.freq)
    
    def fit(
        self, 
        targets: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'DeepProbabilisticForecasting':
        """
        Fit the deep probabilistic forecasting model.
        
        Args:
            targets: Target time series (N x T)
            covariates: Covariate time series (N x T x K)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            **kwargs: Additional arguments
            
        Returns:
            Self (for method chaining)
        """
        # Prepare training dataset
        train_dataset = self._prepare_dataset(
            targets, covariates, start_dates, feat_static_cat
        )
        
        # Create estimator
        self.estimator = create_stoat_estimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            distribution=self.distribution,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            ctx=self.ctx,
            **kwargs
        )
        
        # Train the model
        self.predictor = self.estimator.train(train_dataset)
        
        return self
    
    def predict(
        self, 
        targets: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None
    ) -> List:
        """
        Generate probabilistic forecasts.
        
        Args:
            targets: Target time series (N x T)
            covariates: Covariate time series (N x T x K)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            num_samples: Number of sample paths (overrides default)
            
        Returns:
            List of forecast objects
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare test dataset
        test_dataset = self._prepare_dataset(
            targets, covariates, start_dates, feat_static_cat
        )
        
        # Generate forecasts
        if num_samples is None:
            num_samples = self.num_sample_paths
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,
            predictor=self.predictor,
            num_samples=num_samples
        )
        
        forecasts = list(forecast_it)
        return forecasts
    
    def evaluate(
        self, 
        forecasts: List,
        targets: np.ndarray,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Evaluate forecast performance.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            quantiles: Quantiles for evaluation
            
        Returns:
            Tuple of (aggregate_metrics, item_metrics)
        """
        # Prepare test dataset for evaluation
        test_dataset = self._prepare_dataset(
            targets, start_dates=start_dates, feat_static_cat=feat_static_cat
        )
        
        # Convert to time series objects
        ts_it = iter(test_dataset)
        tss = list(ts_it)
        
        # Create evaluator
        evaluator = Evaluator(quantiles=quantiles)
        
        # Evaluate
        agg_metrics, item_metrics = evaluator(
            iter(tss), 
            iter(forecasts), 
            num_series=len(test_dataset)
        )
        
        return agg_metrics, item_metrics
    
    def get_forecast_summary(self, forecasts: List) -> Dict:
        """
        Get summary statistics from forecasts.
        
        Args:
            forecasts: List of forecast objects
            
        Returns:
            Dictionary with forecast summary statistics
        """
        summary = {}
        
        for i, forecast in enumerate(forecasts):
            summary[f'series_{i}'] = {
                'mean': forecast.mean.tolist(),
                'median': forecast.quantile(0.5).tolist(),
                'std': forecast.std.tolist(),
                'quantile_10': forecast.quantile(0.1).tolist(),
                'quantile_90': forecast.quantile(0.9).tolist(),
                'num_samples': forecast.num_samples,
                'prediction_length': forecast.prediction_length
            }
        
        return summary
    
    def plot_forecasts(
        self, 
        forecasts: List, 
        targets: np.ndarray,
        plot_length: int = 50,
        prediction_intervals: Tuple[float, float] = (50.0, 90.0),
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot forecasts with prediction intervals.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            plot_length: Length of historical data to plot
            prediction_intervals: Prediction intervals to show
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
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
            ax.plot(range(hist_length), target[-hist_length:], 'b-', label='Historical')
            
            # Plot forecast
            forecast.plot(prediction_intervals=prediction_intervals, color='r', ax=ax)
            
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
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'prediction_length': self.prediction_length,
            'context_length': self.context_length,
            'freq': self.freq,
            'distribution': self.distribution,
            'num_cells': self.num_cells,
            'num_sample_paths': self.num_sample_paths,
            'scaling': self.scaling,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'ctx': self.ctx,
            'is_fitted': self.predictor is not None
        }
        
        return info
