"""
Data processing utilities for STOAT.

This module provides data processors for handling epidemic data,
including loading, preprocessing, and formatting for STOAT models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings


class EpidemicDataProcessor:
    """
    Data processor for epidemic data in STOAT.
    
    This class handles loading, preprocessing, and formatting of epidemic data
    for use with STOAT models.
    """
    
    def __init__(
        self,
        target_columns: List[str],
        covariate_columns: Optional[List[str]] = None,
        region_column: Optional[str] = None,
        date_column: str = "date",
        scaling_method: str = "minmax",
        handle_missing: str = "interpolate"
    ):
        """
        Initialize the epidemic data processor.
        
        Args:
            target_columns: List of target variable column names
            covariate_columns: List of covariate column names
            region_column: Name of region identifier column
            date_column: Name of date column
            scaling_method: Scaling method ('minmax', 'standard', 'none')
            handle_missing: Method for handling missing values ('interpolate', 'drop', 'fill')
        """
        self.target_columns = target_columns
        self.covariate_columns = covariate_columns or []
        self.region_column = region_column
        self.date_column = date_column
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        
        # Scalers for different data types
        self.target_scaler = None
        self.covariate_scaler = None
        
        # Data metadata
        self.regions = None
        self.dates = None
        self.n_regions = None
        self.n_timepoints = None
        
    def load_and_preprocess(
        self,
        data_path: str,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Load and preprocess epidemic data.
        
        Args:
            data_path: Path to the data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary with processed data
        """
        # Load data
        data = self._load_data(data_path, **kwargs)
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        return processed_data
    
    def _load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to the data file
            **kwargs: Additional arguments
            
        Returns:
            Loaded DataFrame
        """
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path, **kwargs)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data = pd.read_excel(data_path, **kwargs)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Dictionary with processed data
        """
        # Handle date column
        if self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column])
            data = data.sort_values(self.date_column)
            self.dates = data[self.date_column].values
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Extract regions
        if self.region_column and self.region_column in data.columns:
            self.regions = data[self.region_column].unique()
            self.n_regions = len(self.regions)
        else:
            # Assume each column represents a region
            self.regions = self.target_columns
            self.n_regions = len(self.target_columns)
        
        # Extract target data
        targets = self._extract_targets(data)
        
        # Extract covariate data
        covariates = self._extract_covariates(data)
        
        # Scale data
        if self.scaling_method != "none":
            targets, covariates = self._scale_data(targets, covariates)
        
        # Prepare output
        processed_data = {
            'targets': targets,
            'regions': self.regions,
            'dates': self.dates
        }
        
        if covariates is not None:
            processed_data['covariates'] = covariates
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        if self.handle_missing == "interpolate":
            # Interpolate missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].interpolate(method='linear')
            
            # Fill remaining missing values with forward fill
            data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
            
        elif self.handle_missing == "drop":
            # Drop rows with missing values
            data = data.dropna()
            
        elif self.handle_missing == "fill":
            # Fill with zeros
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(0)
        
        return data
    
    def _extract_targets(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract target variables from data.
        
        Args:
            data: DataFrame
            
        Returns:
            Target array (N x T)
        """
        if self.region_column and self.region_column in data.columns:
            # Long format: pivot to wide format
            target_data = data.pivot(
                index=self.date_column,
                columns=self.region_column,
                values=self.target_columns[0]
            )
            targets = target_data.values.T  # (N x T)
        else:
            # Wide format: extract columns directly
            targets = data[self.target_columns].values.T  # (N x T)
        
        self.n_timepoints = targets.shape[1]
        
        return targets
    
    def _extract_covariates(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract covariate variables from data.
        
        Args:
            data: DataFrame
            
        Returns:
            Covariate array (N x T x K) or None
        """
        if not self.covariate_columns:
            return None
        
        if self.region_column and self.region_column in data.columns:
            # Long format: pivot each covariate
            covariates_list = []
            for cov_col in self.covariate_columns:
                cov_data = data.pivot(
                    index=self.date_column,
                    columns=self.region_column,
                    values=cov_col
                )
                covariates_list.append(cov_data.values.T)
            
            # Stack covariates: (K x N x T) -> (N x T x K)
            covariates = np.stack(covariates_list, axis=-1)
        else:
            # Wide format: extract columns directly
            cov_data = data[self.covariate_columns].values
            # Reshape to (N x T x K) - assuming data is organized by time
            n_covariates = len(self.covariate_columns)
            n_timepoints = len(data)
            n_regions = self.n_regions
            
            covariates = cov_data.reshape(n_regions, n_timepoints, n_covariates)
        
        return covariates
    
    def _scale_data(
        self, 
        targets: np.ndarray, 
        covariates: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale the data using the specified method.
        
        Args:
            targets: Target array (N x T)
            covariates: Covariate array (N x T x K)
            
        Returns:
            Tuple of (scaled_targets, scaled_covariates)
        """
        # Scale targets
        if self.scaling_method == "minmax":
            self.target_scaler = MinMaxScaler()
        elif self.scaling_method == "standard":
            self.target_scaler = StandardScaler()
        else:
            return targets, covariates
        
        # Reshape for scaling: (N x T) -> (N*T, 1) -> (N x T)
        targets_reshaped = targets.T.reshape(-1, 1)
        targets_scaled = self.target_scaler.fit_transform(targets_reshaped)
        targets_scaled = targets_scaled.reshape(-1, targets.shape[0]).T
        
        # Scale covariates
        covariates_scaled = None
        if covariates is not None:
            if self.scaling_method == "minmax":
                self.covariate_scaler = MinMaxScaler()
            elif self.scaling_method == "standard":
                self.covariate_scaler = StandardScaler()
            
            # Reshape for scaling: (N x T x K) -> (N*T*K, 1) -> (N x T x K)
            N, T, K = covariates.shape
            covariates_reshaped = covariates.reshape(-1, 1)
            covariates_scaled = self.covariate_scaler.fit_transform(covariates_reshaped)
            covariates_scaled = covariates_scaled.reshape(N, T, K)
        
        return targets_scaled, covariates_scaled
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets.
        
        Args:
            targets: Scaled targets
            
        Returns:
            Original scale targets
        """
        if self.target_scaler is None:
            return targets
        
        # Reshape for inverse transform
        targets_reshaped = targets.T.reshape(-1, 1)
        targets_original = self.target_scaler.inverse_transform(targets_reshaped)
        targets_original = targets_original.reshape(-1, targets.shape[0]).T
        
        return targets_original
    
    def inverse_transform_covariates(self, covariates: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled covariates.
        
        Args:
            covariates: Scaled covariates
            
        Returns:
            Original scale covariates
        """
        if self.covariate_scaler is None:
            return covariates
        
        # Reshape for inverse transform
        N, T, K = covariates.shape
        covariates_reshaped = covariates.reshape(-1, 1)
        covariates_original = self.covariate_scaler.inverse_transform(covariates_reshaped)
        covariates_original = covariates_original.reshape(N, T, K)
        
        return covariates_original
    
    def create_train_test_split(
        self,
        targets: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create train-test split for the data.
        
        Args:
            targets: Target array (N x T)
            covariates: Covariate array (N x T x K)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with train/test splits
        """
        N, T = targets.shape
        test_length = int(T * test_size)
        train_length = T - test_length
        
        # Split targets
        train_targets = targets[:, :train_length]
        test_targets = targets[:, train_length:]
        
        result = {
            'train_targets': train_targets,
            'test_targets': test_targets
        }
        
        # Split covariates if provided
        if covariates is not None:
            train_covariates = covariates[:, :train_length, :]
            test_covariates = covariates[:, train_length:, :]
            
            result['train_covariates'] = train_covariates
            result['test_covariates'] = test_covariates
        
        return result
    
    def get_data_info(self) -> Dict:
        """
        Get information about the processed data.
        
        Returns:
            Dictionary with data information
        """
        info = {
            'n_regions': self.n_regions,
            'n_timepoints': self.n_timepoints,
            'regions': self.regions.tolist() if self.regions is not None else None,
            'target_columns': self.target_columns,
            'covariate_columns': self.covariate_columns,
            'scaling_method': self.scaling_method,
            'handle_missing': self.handle_missing
        }
        
        return info
