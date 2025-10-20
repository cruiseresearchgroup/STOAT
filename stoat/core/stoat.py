"""
Main STOAT class that integrates spatial causal inference and deep probabilistic forecasting.

This module provides the main interface for the STOAT framework, combining
spatial causal inference with deep probabilistic forecasting for epidemic prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .spatial_causal import SpatialCausalInference
from .forecasting import DeepProbabilisticForecasting


class STOAT:
    """
    STOAT: Spatial-Temporal Causal Inference for Epidemic Forecasting.
    
    This is the main class that integrates spatial causal inference with deep
    probabilistic forecasting to provide comprehensive epidemic forecasting
    with uncertainty quantification and causal interpretability.
    """
    
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        spatial_matrix: np.ndarray,
        freq: str = "D",
        distribution: str = "laplace",
        num_cells: int = 64,
        num_sample_paths: int = 100,
        scaling: bool = True,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        ctx: str = "cpu",
        causal_method: str = "2sls",
        treatment_indicator: Optional[np.ndarray] = None,
        post_treatment_period: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize STOAT model.
        
        Args:
            prediction_length: Length of prediction horizon
            context_length: Length of context window
            spatial_matrix: Spatial relation matrix (N x N)
            freq: Frequency of the time series
            distribution: Distribution type ('gaussian', 'laplace', 'student_t')
            num_cells: Number of LSTM cells
            num_sample_paths: Number of sample paths for prediction
            scaling: Whether to use scaling
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            ctx: Context (device) for training
            causal_method: Causal inference method ('2sls', 'ols')
            treatment_indicator: Binary treatment indicator (N,)
            post_treatment_period: Post-treatment period indicator (T,)
            **kwargs: Additional arguments
        """
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.spatial_matrix = spatial_matrix
        self.freq = freq
        self.distribution = distribution
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ctx = ctx
        self.causal_method = causal_method
        self.treatment_indicator = treatment_indicator
        self.post_treatment_period = post_treatment_period
        
        # Initialize modules
        self.spatial_causal = SpatialCausalInference(
            spatial_matrix=spatial_matrix,
            treatment_indicator=treatment_indicator,
            post_treatment_period=post_treatment_period,
            method=causal_method
        )
        
        self.forecasting = DeepProbabilisticForecasting(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            distribution=distribution,
            num_cells=num_cells,
            num_sample_paths=num_sample_paths,
            scaling=scaling,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            ctx=ctx,
            **kwargs
        )
        
        # Store fitted state
        self.is_fitted = False
        self.causal_parameters = None
        self.forecast_parameters = None
        
    def fit(
        self,
        targets: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None,
        fit_causal: bool = True,
        **kwargs
    ) -> 'STOAT':
        """
        Fit the STOAT model.
        
        Args:
            targets: Target time series (N x T)
            covariates: Covariate time series (N x T x K)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            fit_causal: Whether to fit the causal inference module
            **kwargs: Additional arguments
            
        Returns:
            Self (for method chaining)
        """
        if targets.ndim != 2:
            raise ValueError("targets must be a 2D array with shape (N, T)")
        
        N, T = targets.shape
        if N != self.spatial_matrix.shape[0]:
            raise ValueError("Number of regions in targets must match spatial matrix")
        
        # Step 1: Fit spatial causal inference module
        if fit_causal:
            print("Fitting spatial causal inference module...")
            self.spatial_causal.covariates = covariates
            self.spatial_causal.fit(targets)
            self.causal_parameters = self.spatial_causal.get_parameters()
            
            # Get causally adjusted targets
            targets_adjusted = self.spatial_causal.causal_adjustment(targets)
            
            # Get spatially adjusted inputs
            z_adjusted = self.spatial_causal.spatial_adjustment(targets_adjusted)
            
            print("Spatial causal inference completed.")
            print(f"Treatment effect (δ): {self.causal_parameters.get('delta', 'N/A')}")
            print(f"Spatial coefficient (ρ): {self.causal_parameters.get('rho', 'N/A')}")
        else:
            # Use original targets if causal inference is skipped
            z_adjusted = targets
            warnings.warn("Causal inference skipped. Using original targets.")
        
        # Step 2: Fit deep probabilistic forecasting module
        print("Fitting deep probabilistic forecasting module...")
        self.forecasting.fit(
            targets=z_adjusted,
            covariates=covariates,
            start_dates=start_dates,
            feat_static_cat=feat_static_cat,
            **kwargs
        )
        self.forecast_parameters = self.forecasting.get_model_info()
        
        print("Deep probabilistic forecasting completed.")
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        targets: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
        apply_causal_adjustment: bool = True
    ) -> List:
        """
        Generate probabilistic forecasts.
        
        Args:
            targets: Target time series (N x T)
            covariates: Covariate time series (N x T x K)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            num_samples: Number of sample paths
            apply_causal_adjustment: Whether to apply causal adjustment
            
        Returns:
            List of forecast objects
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply causal adjustment if requested and available
        if apply_causal_adjustment and self.causal_parameters is not None:
            targets_adjusted = self.spatial_causal.causal_adjustment(targets)
            z_adjusted = self.spatial_causal.spatial_adjustment(targets_adjusted)
        else:
            z_adjusted = targets
        
        # Generate forecasts
        forecasts = self.forecasting.predict(
            targets=z_adjusted,
            covariates=covariates,
            start_dates=start_dates,
            feat_static_cat=feat_static_cat,
            num_samples=num_samples
        )
        
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
        return self.forecasting.evaluate(
            forecasts=forecasts,
            targets=targets,
            start_dates=start_dates,
            feat_static_cat=feat_static_cat,
            quantiles=quantiles
        )
    
    def get_causal_interpretation(self) -> Dict:
        """
        Get interpretable insights from the causal inference module.
        
        Returns:
            Dictionary with causal interpretation
        """
        if self.causal_parameters is None:
            warnings.warn("Causal inference module not fitted.")
            return {}
        
        return self.spatial_causal.get_interpretation()
    
    def get_forecast_summary(self, forecasts: List) -> Dict:
        """
        Get summary statistics from forecasts.
        
        Args:
            forecasts: List of forecast objects
            
        Returns:
            Dictionary with forecast summary statistics
        """
        return self.forecasting.get_forecast_summary(forecasts)
    
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
        self.forecasting.plot_forecasts(
            forecasts=forecasts,
            targets=targets,
            plot_length=plot_length,
            prediction_intervals=prediction_intervals,
            figsize=figsize
        )
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'model_type': 'STOAT',
            'is_fitted': self.is_fitted,
            'prediction_length': self.prediction_length,
            'context_length': self.context_length,
            'freq': self.freq,
            'distribution': self.distribution,
            'causal_method': self.causal_method,
            'spatial_matrix_shape': self.spatial_matrix.shape,
            'causal_parameters': self.causal_parameters,
            'forecast_parameters': self.forecast_parameters
        }
        
        if self.causal_parameters is not None:
            summary['causal_interpretation'] = self.get_causal_interpretation()
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        model_data = {
            'causal_parameters': self.causal_parameters,
            'forecast_parameters': self.forecast_parameters,
            'spatial_matrix': self.spatial_matrix,
            'treatment_indicator': self.treatment_indicator,
            'post_treatment_period': self.post_treatment_period,
            'config': {
                'prediction_length': self.prediction_length,
                'context_length': self.context_length,
                'freq': self.freq,
                'distribution': self.distribution,
                'causal_method': self.causal_method,
                'num_cells': self.num_cells,
                'num_sample_paths': self.num_sample_paths,
                'scaling': self.scaling,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'ctx': self.ctx
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'STOAT':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded STOAT model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        config = model_data['config']
        model = cls(
            prediction_length=config['prediction_length'],
            context_length=config['context_length'],
            spatial_matrix=model_data['spatial_matrix'],
            freq=config['freq'],
            distribution=config['distribution'],
            causal_method=config['causal_method'],
            treatment_indicator=model_data['treatment_indicator'],
            post_treatment_period=model_data['post_treatment_period'],
            num_cells=config['num_cells'],
            num_sample_paths=config['num_sample_paths'],
            scaling=config['scaling'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            ctx=config['ctx']
        )
        
        # Restore fitted state
        model.causal_parameters = model_data['causal_parameters']
        model.forecast_parameters = model_data['forecast_parameters']
        model.is_fitted = True
        
        return model
