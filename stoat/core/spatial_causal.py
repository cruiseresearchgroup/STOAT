"""
Spatial Causal Inference Module for STOAT.

This module implements the spatial causal inference mechanism that extends
the classical Difference-in-Differences (DiD) framework by incorporating
spatial dependencies through spatial relation matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings


class SpatialCausalInference:
    """
    Spatial Causal Inference module for STOAT.
    
    This class implements the spatially-enhanced DiD framework that captures
    the complex interplay between spatial dependencies and causal effects
    in epidemic dynamics.
    """
    
    def __init__(
        self,
        spatial_matrix: np.ndarray,
        treatment_indicator: Optional[np.ndarray] = None,
        post_treatment_period: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
        method: str = "2sls"
    ):
        """
        Initialize the spatial causal inference module.
        
        Args:
            spatial_matrix: Spatial relation matrix S (N x N)
            treatment_indicator: Binary treatment indicator T_i
            post_treatment_period: Post-treatment period indicator Post_t
            covariates: Multi-dimensional covariate matrix c_{i,t}
            method: Estimation method ('2sls', 'ols', 'ml')
        """
        self.spatial_matrix = spatial_matrix
        self.treatment_indicator = treatment_indicator
        self.post_treatment_period = post_treatment_period
        self.covariates = covariates
        self.method = method
        
        # Parameters to be estimated
        self.rho = None  # Spatial autoregressive coefficient
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Treatment effect
        self.beta_2 = None  # Post-treatment effect
        self.delta = None  # Treatment effect (main parameter of interest)
        self.gamma = None  # Covariate effects
        
        # Standard errors and statistics
        self.std_errors = {}
        self.t_stats = {}
        self.p_values = {}
        
        # Fitted values and residuals
        self.fitted_values = None
        self.residuals = None
        
    def _compute_spatial_lag(self, y: np.ndarray) -> np.ndarray:
        """
        Compute spatial lag term: S * y
        
        Args:
            y: Outcome variable (N x T)
            
        Returns:
            Spatial lag term (N x T)
        """
        return np.dot(self.spatial_matrix, y)
    
    def _create_design_matrix(
        self, 
        y: np.ndarray, 
        spatial_lag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create design matrix for the spatial DiD model.
        
        Args:
            y: Outcome variable (N x T)
            spatial_lag: Spatial lag term (N x T)
            
        Returns:
            X: Design matrix
            y_vec: Vectorized outcome variable
        """
        N, T = y.shape
        
        # Vectorize data
        y_vec = y.flatten()
        spatial_lag_vec = spatial_lag.flatten()
        
        # Create design matrix
        X_list = []
        
        # Intercept
        X_list.append(np.ones(N * T))
        
        # Treatment indicator (if provided)
        if self.treatment_indicator is not None:
            treatment_vec = np.tile(self.treatment_indicator, T)
            X_list.append(treatment_vec)
        
        # Post-treatment period (if provided)
        if self.post_treatment_period is not None:
            post_vec = np.repeat(self.post_treatment_period, N)
            X_list.append(post_vec)
        
        # Treatment interaction (if both treatment and post-treatment provided)
        if (self.treatment_indicator is not None and 
            self.post_treatment_period is not None):
            interaction_vec = np.tile(self.treatment_indicator, T) * np.repeat(self.post_treatment_period, N)
            X_list.append(interaction_vec)
        
        # Covariates (if provided)
        if self.covariates is not None:
            if self.covariates.ndim == 3:  # (N, T, K)
                cov_vec = self.covariates.reshape(-1, self.covariates.shape[-1])
                X_list.extend([cov_vec[:, k] for k in range(cov_vec.shape[1])])
            elif self.covariates.ndim == 2:  # (N*T, K)
                X_list.extend([self.covariates[:, k] for k in range(self.covariates.shape[1])])
        
        # Spatial lag (will be added separately for 2SLS)
        if self.method != "2sls":
            X_list.append(spatial_lag_vec)
        
        X = np.column_stack(X_list)
        
        return X, y_vec
    
    def _estimate_2sls(self, y: np.ndarray) -> Dict:
        """
        Estimate parameters using two-stage least squares (2SLS).
        
        Args:
            y: Outcome variable (N x T)
            
        Returns:
            Dictionary with estimation results
        """
        N, T = y.shape
        
        # First stage: regress spatial lag on instruments
        spatial_lag = self._compute_spatial_lag(y)
        
        # Use spatial matrix as instruments (common in spatial econometrics)
        instruments = self.spatial_matrix
        
        # Create instrument matrix
        Z_list = []
        for t in range(T):
            Z_t = np.column_stack([
                np.ones(N),
                instruments,
                # Add other instruments if available
            ])
            Z_list.append(Z_t)
        
        Z = np.vstack(Z_list)
        spatial_lag_vec = spatial_lag.flatten()
        
        # First stage regression
        first_stage = LinearRegression()
        first_stage.fit(Z, spatial_lag_vec)
        spatial_lag_pred = first_stage.predict(Z)
        
        # Second stage: use predicted spatial lag
        X, y_vec = self._create_design_matrix(y, spatial_lag_pred.reshape(N, T))
        
        # Add predicted spatial lag to design matrix
        X = np.column_stack([X, spatial_lag_pred])
        
        # Second stage regression
        second_stage = LinearRegression()
        second_stage.fit(X, y_vec)
        
        # Extract parameters
        coef = second_stage.coef_
        intercept = second_stage.intercept_
        
        # Organize results
        results = {
            'rho': coef[-1],  # Spatial autoregressive coefficient
            'beta_0': intercept,
            'coefficients': coef[:-1],
            'fitted_values': second_stage.predict(X),
            'residuals': y_vec - second_stage.predict(X),
            'r_squared': second_stage.score(X, y_vec)
        }
        
        return results
    
    def _estimate_ols(self, y: np.ndarray) -> Dict:
        """
        Estimate parameters using ordinary least squares (OLS).
        
        Args:
            y: Outcome variable (N x T)
            
        Returns:
            Dictionary with estimation results
        """
        spatial_lag = self._compute_spatial_lag(y)
        X, y_vec = self._create_design_matrix(y, spatial_lag)
        
        # Add spatial lag to design matrix
        X = np.column_stack([X, spatial_lag.flatten()])
        
        # OLS regression
        ols = LinearRegression()
        ols.fit(X, y_vec)
        
        # Extract parameters
        coef = ols.coef_
        intercept = ols.intercept_
        
        # Organize results
        results = {
            'rho': coef[-1],  # Spatial autoregressive coefficient
            'beta_0': intercept,
            'coefficients': coef[:-1],
            'fitted_values': ols.predict(X),
            'residuals': y_vec - ols.predict(X),
            'r_squared': ols.score(X, y_vec)
        }
        
        return results
    
    def fit(self, y: np.ndarray) -> 'SpatialCausalInference':
        """
        Fit the spatial causal inference model.
        
        Args:
            y: Outcome variable (N x T)
            
        Returns:
            Self (for method chaining)
        """
        if y.ndim != 2:
            raise ValueError("y must be a 2D array with shape (N, T)")
        
        N, T = y.shape
        if N != self.spatial_matrix.shape[0]:
            raise ValueError("Number of regions in y must match spatial matrix")
        
        # Estimate parameters based on method
        if self.method == "2sls":
            results = self._estimate_2sls(y)
        elif self.method == "ols":
            results = self._estimate_ols(y)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Store results
        self.rho = results['rho']
        self.beta_0 = results['beta_0']
        self.fitted_values = results['fitted_values']
        self.residuals = results['residuals']
        
        # Extract other parameters based on what's available
        coef_idx = 0
        if self.treatment_indicator is not None:
            self.beta_1 = results['coefficients'][coef_idx]
            coef_idx += 1
        
        if self.post_treatment_period is not None:
            self.beta_2 = results['coefficients'][coef_idx]
            coef_idx += 1
        
        if (self.treatment_indicator is not None and 
            self.post_treatment_period is not None):
            self.delta = results['coefficients'][coef_idx]
            coef_idx += 1
        
        if self.covariates is not None:
            n_covariates = (self.covariates.shape[-1] if self.covariates.ndim == 3 
                          else self.covariates.shape[1])
            self.gamma = results['coefficients'][coef_idx:coef_idx + n_covariates]
        
        return self
    
    def causal_adjustment(self, y: np.ndarray) -> np.ndarray:
        """
        Perform causal adjustment to remove treatment effects.
        
        Args:
            y: Outcome variable (N x T)
            
        Returns:
            Causally adjusted outcome variable
        """
        if self.delta is None:
            warnings.warn("Treatment effect not estimated. Returning original data.")
            return y
        
        N, T = y.shape
        y_adjusted = y.copy()
        
        # Remove treatment effects
        if (self.treatment_indicator is not None and 
            self.post_treatment_period is not None):
            for i in range(N):
                for t in range(T):
                    if (self.treatment_indicator[i] == 1 and 
                        self.post_treatment_period[t] == 1):
                        y_adjusted[i, t] -= self.delta
        
        return y_adjusted
    
    def spatial_adjustment(self, y_adjusted: np.ndarray) -> np.ndarray:
        """
        Apply spatial adjustment using estimated spatial effects.
        
        Args:
            y_adjusted: Causally adjusted outcome variable
            
        Returns:
            Spatially adjusted input for forecasting
        """
        if self.rho is None:
            warnings.warn("Spatial coefficient not estimated. Returning causally adjusted data.")
            return y_adjusted
        
        # Compute spatial lag
        spatial_lag = self._compute_spatial_lag(y_adjusted)
        
        # Apply spatial adjustment
        z = y_adjusted + self.rho * spatial_lag
        
        return z
    
    def get_parameters(self) -> Dict:
        """
        Get estimated parameters.
        
        Returns:
            Dictionary with estimated parameters
        """
        params = {}
        if self.rho is not None:
            params['rho'] = self.rho
        if self.beta_0 is not None:
            params['beta_0'] = self.beta_0
        if self.beta_1 is not None:
            params['beta_1'] = self.beta_1
        if self.beta_2 is not None:
            params['beta_2'] = self.beta_2
        if self.delta is not None:
            params['delta'] = self.delta
        if self.gamma is not None:
            params['gamma'] = self.gamma
        
        return params
    
    def get_interpretation(self) -> Dict:
        """
        Get interpretable insights from the estimated parameters.
        
        Returns:
            Dictionary with interpretable insights
        """
        insights = {}
        
        if self.delta is not None:
            if self.delta < -0.1:
                insights['treatment_effect'] = "Strong negative effect - interventions effectively reduce cases"
            elif self.delta > 0.1:
                insights['treatment_effect'] = "Positive effect - interventions may increase cases"
            else:
                insights['treatment_effect'] = "Limited intervention effectiveness"
        
        if self.rho is not None:
            if abs(self.rho) > 0.3:
                insights['spatial_spillover'] = "Strong spatial correlation - neighboring regions significantly influence local outcomes"
            else:
                insights['spatial_spillover'] = "Weak spatial correlation - limited cross-regional influence"
        
        if self.gamma is not None:
            covariate_names = ['reproduction_rate', 'stringency_index', 'vaccination_share', 'icu_patients']
            for i, gamma_i in enumerate(self.gamma):
                if i < len(covariate_names):
                    if gamma_i > 0:
                        insights[f'{covariate_names[i]}_effect'] = f"Positive effect - higher values increase cases"
                    else:
                        insights[f'{covariate_names[i]}_effect'] = f"Negative effect - higher values decrease cases"
        
        return insights
