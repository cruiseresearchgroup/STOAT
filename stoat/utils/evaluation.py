"""
Evaluation utilities for STOAT.

This module provides comprehensive evaluation metrics and utilities for
assessing the performance of STOAT models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from gluonts.evaluation import Evaluator
import warnings


class STOATEvaluator:
    """
    Comprehensive evaluator for STOAT models.
    
    This class provides various evaluation metrics for both forecasting
    performance and causal inference quality.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        seasonal_period: Optional[int] = None
    ):
        """
        Initialize STOAT evaluator.
        
        Args:
            quantiles: Quantiles for evaluation
            seasonal_period: Seasonal period for seasonal error calculation
        """
        self.quantiles = quantiles
        self.seasonal_period = seasonal_period
        self.gluon_evaluator = Evaluator(quantiles=quantiles)
    
    def evaluate_forecasts(
        self,
        forecasts: List,
        targets: np.ndarray,
        start_dates: Optional[List[pd.Timestamp]] = None,
        feat_static_cat: Optional[np.ndarray] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Evaluate forecasting performance.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            start_dates: List of start dates for each series
            feat_static_cat: Static categorical features
            
        Returns:
            Tuple of (aggregate_metrics, item_metrics)
        """
        # Prepare test dataset for evaluation
        from ..data.loaders import prepare_dataset
        
        test_dataset = prepare_dataset(
            targets, start_dates=start_dates, feat_static_cat=feat_static_cat
        )
        
        # Convert to time series objects
        ts_it = iter(test_dataset)
        tss = list(ts_it)
        
        # Evaluate using GluonTS evaluator
        agg_metrics, item_metrics = self.gluon_evaluator(
            iter(tss), 
            iter(forecasts), 
            num_series=len(test_dataset)
        )
        
        return agg_metrics, item_metrics
    
    def evaluate_causal_inference(
        self,
        causal_parameters: Dict,
        true_parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate causal inference quality.
        
        Args:
            causal_parameters: Estimated causal parameters
            true_parameters: True parameters (if available)
            
        Returns:
            Dictionary with causal evaluation metrics
        """
        evaluation = {}
        
        # Parameter significance
        if 'delta' in causal_parameters:
            delta = causal_parameters['delta']
            if abs(delta) > 0.1:
                evaluation['treatment_significance'] = 'Significant'
            else:
                evaluation['treatment_significance'] = 'Not significant'
        
        if 'rho' in causal_parameters:
            rho = causal_parameters['rho']
            if abs(rho) > 0.3:
                evaluation['spatial_significance'] = 'Strong spatial correlation'
            else:
                evaluation['spatial_significance'] = 'Weak spatial correlation'
        
        # Compare with true parameters if available
        if true_parameters is not None:
            for param_name in ['delta', 'rho', 'gamma']:
                if param_name in causal_parameters and param_name in true_parameters:
                    estimated = causal_parameters[param_name]
                    true_val = true_parameters[param_name]
                    
                    if isinstance(estimated, np.ndarray) and isinstance(true_val, np.ndarray):
                        mse = np.mean((estimated - true_val) ** 2)
                        mae = np.mean(np.abs(estimated - true_val))
                        evaluation[f'{param_name}_mse'] = mse
                        evaluation[f'{param_name}_mae'] = mae
                    else:
                        error = abs(estimated - true_val)
                        evaluation[f'{param_name}_error'] = error
        
        return evaluation
    
    def compute_forecast_metrics(
        self,
        forecasts: List,
        targets: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive forecast metrics.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            
        Returns:
            Dictionary with forecast metrics
        """
        metrics = {}
        
        # Extract forecast values
        forecast_means = []
        forecast_medians = []
        forecast_stds = []
        
        for forecast in forecasts:
            forecast_means.append(forecast.mean)
            forecast_medians.append(forecast.quantile(0.5))
            forecast_std = np.std(forecast.samples, axis=0)
            forecast_stds.append(forecast_std)
        
        forecast_means = np.array(forecast_means)
        forecast_medians = np.array(forecast_medians)
        forecast_stds = np.array(forecast_stds)
        
        # Extract true values (last prediction_length values)
        true_values = targets[:, -self.prediction_length:]
        
        # Point forecast metrics
        metrics['mse'] = np.mean((forecast_means - true_values) ** 2)
        metrics['mae'] = np.mean(np.abs(forecast_means - true_values))
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mape'] = np.mean(np.abs((true_values - forecast_means) / true_values)) * 100
        
        # Median forecast metrics
        metrics['median_mse'] = np.mean((forecast_medians - true_values) ** 2)
        metrics['median_mae'] = np.mean(np.abs(forecast_medians - true_values))
        
        # Uncertainty metrics
        metrics['mean_std'] = np.mean(forecast_stds)
        metrics['std_std'] = np.std(forecast_stds)
        
        # Coverage metrics
        for q in self.quantiles:
            if q != 0.5:  # Skip median
                lower_q = (1 - q) / 2
                upper_q = (1 + q) / 2
                
                lower_bounds = [f.quantile(lower_q) for f in forecasts]
                upper_bounds = [f.quantile(upper_q) for f in forecasts]
                
                lower_bounds = np.array(lower_bounds)
                upper_bounds = np.array(upper_bounds)
                
                coverage = np.mean(
                    (true_values >= lower_bounds) & (true_values <= upper_bounds)
                )
                metrics[f'coverage_{int(q*100)}'] = coverage
        
        return metrics
    
    def compute_spatial_metrics(
        self,
        forecasts: List,
        targets: np.ndarray,
        spatial_matrix: np.ndarray
    ) -> Dict:
        """
        Compute spatial-specific evaluation metrics.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            spatial_matrix: Spatial relation matrix
            
        Returns:
            Dictionary with spatial metrics
        """
        metrics = {}
        
        # Extract forecast means
        forecast_means = np.array([f.mean for f in forecasts])
        true_values = targets[:, -self.prediction_length:]
        
        # Spatial correlation of errors
        errors = forecast_means - true_values
        spatial_errors = np.dot(spatial_matrix, errors)
        
        # Compute spatial correlation
        spatial_corr = np.corrcoef(errors.flatten(), spatial_errors.flatten())[0, 1]
        metrics['spatial_error_correlation'] = spatial_corr
        
        # Spatial autocorrelation of forecasts
        forecast_autocorr = np.corrcoef(forecast_means.flatten(), 
                                       np.dot(spatial_matrix, forecast_means).flatten())[0, 1]
        metrics['forecast_spatial_autocorr'] = forecast_autocorr
        
        # Spatial autocorrelation of true values
        true_autocorr = np.corrcoef(true_values.flatten(), 
                                   np.dot(spatial_matrix, true_values).flatten())[0, 1]
        metrics['true_spatial_autocorr'] = true_autocorr
        
        return metrics
    
    def evaluate_model_interpretability(
        self,
        causal_parameters: Dict,
        forecast_parameters: Dict
    ) -> Dict:
        """
        Evaluate model interpretability.
        
        Args:
            causal_parameters: Causal inference parameters
            forecast_parameters: Forecasting parameters
            
        Returns:
            Dictionary with interpretability metrics
        """
        interpretability = {}
        
        # Causal interpretability
        if 'delta' in causal_parameters:
            delta = causal_parameters['delta']
            if delta < -0.1:
                interpretability['treatment_interpretation'] = 'Strong negative effect'
            elif delta > 0.1:
                interpretability['treatment_interpretation'] = 'Strong positive effect'
            else:
                interpretability['treatment_interpretation'] = 'Weak effect'
        
        if 'rho' in causal_parameters:
            rho = causal_parameters['rho']
            if abs(rho) > 0.3:
                interpretability['spatial_interpretation'] = 'Strong spatial spillovers'
            else:
                interpretability['spatial_interpretation'] = 'Weak spatial spillovers'
        
        # Model complexity
        if 'num_cells' in forecast_parameters:
            num_cells = forecast_parameters['num_cells']
            if num_cells > 100:
                interpretability['model_complexity'] = 'High'
            elif num_cells > 50:
                interpretability['model_complexity'] = 'Medium'
            else:
                interpretability['model_complexity'] = 'Low'
        
        return interpretability
    
    def generate_evaluation_report(
        self,
        forecasts: List,
        targets: np.ndarray,
        causal_parameters: Optional[Dict] = None,
        forecast_parameters: Optional[Dict] = None,
        spatial_matrix: Optional[np.ndarray] = None,
        true_parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
            causal_parameters: Causal inference parameters
            forecast_parameters: Forecasting parameters
            spatial_matrix: Spatial relation matrix
            true_parameters: True parameters (if available)
            
        Returns:
            Comprehensive evaluation report
        """
        report = {}
        
        # Forecast evaluation
        report['forecast_metrics'] = self.compute_forecast_metrics(forecasts, targets)
        
        # Causal inference evaluation
        if causal_parameters is not None:
            report['causal_evaluation'] = self.evaluate_causal_inference(
                causal_parameters, true_parameters
            )
        
        # Spatial evaluation
        if spatial_matrix is not None:
            report['spatial_metrics'] = self.compute_spatial_metrics(
                forecasts, targets, spatial_matrix
            )
        
        # Interpretability evaluation
        if causal_parameters is not None and forecast_parameters is not None:
            report['interpretability'] = self.evaluate_model_interpretability(
                causal_parameters, forecast_parameters
            )
        
        return report
    
    def plot_evaluation_results(
        self,
        forecasts: List,
        targets: np.ndarray,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot comprehensive evaluation results.
        
        Args:
            forecasts: List of forecast objects
            targets: True target values (N x T)
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
            hist_length = min(50, len(target))
            ax.plot(range(hist_length), target[-hist_length:], 'b-', label='Historical', linewidth=2)
            
            # Plot forecast with prediction intervals
            forecast.plot(prediction_intervals=(50.0, 90.0), color='r', ax=ax)
            
            # Add true values for comparison
            true_future = target[-self.prediction_length:]
            ax.plot(range(hist_length, hist_length + len(true_future)), 
                   true_future, 'g--', label='True', linewidth=2)
            
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
