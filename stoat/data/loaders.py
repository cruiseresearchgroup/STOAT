"""
Data loading utilities for STOAT.

This module provides data loaders for common epidemic datasets and
utilities for preparing data in the format required by STOAT models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import warnings


def load_epidemic_data(
    data_path: str,
    target_columns: List[str],
    covariate_columns: Optional[List[str]] = None,
    region_column: Optional[str] = None,
    date_column: str = "date",
    scaling_method: str = "minmax",
    **kwargs
) -> Dict[str, Any]:
    """
    Load epidemic data using the EpidemicDataProcessor.
    
    Args:
        data_path: Path to the data file
        target_columns: List of target variable column names
        covariate_columns: List of covariate column names
        region_column: Name of region identifier column
        date_column: Name of date column
        scaling_method: Scaling method ('minmax', 'standard', 'none')
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with processed data
    """
    from .processors import EpidemicDataProcessor
    
    processor = EpidemicDataProcessor(
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        region_column=region_column,
        date_column=date_column,
        scaling_method=scaling_method
    )
    
    return processor.load_and_preprocess(data_path, **kwargs)


def prepare_dataset(
    targets: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    start_dates: Optional[List[pd.Timestamp]] = None,
    feat_static_cat: Optional[np.ndarray] = None,
    freq: str = "D"
) -> ListDataset:
    """
    Prepare dataset in GluonTS format.
    
    Args:
        targets: Target time series (N x T)
        covariates: Covariate time series (N x T x K)
        start_dates: List of start dates for each series
        feat_static_cat: Static categorical features
        freq: Frequency of the time series
        
    Returns:
        GluonTS ListDataset
    """
    N, T = targets.shape
    
    if start_dates is None:
        start_dates = [pd.Timestamp('2020-01-01', freq=freq) for _ in range(N)]
    
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
    
    return ListDataset(dataset_entries, freq=freq)


def load_covid_data(
    data_path: str = "data/6_countries_data_with_omicron.csv",
    countries: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load COVID-19 data from the 6 countries dataset.
    
    Args:
        data_path: Path to the COVID data file
        countries: List of countries to include
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Dictionary with processed COVID data
    """
    # Load raw data
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Filter by date range
    if start_date:
        data = data[data['date'] >= start_date]
    if end_date:
        data = data[data['date'] <= end_date]
    
    # Define target and covariate columns
    target_columns = []
    covariate_columns = []
    
    # Extract country names from columns
    if countries is None:
        countries = ['Canada', 'France', 'Italy', 'Spain', 'UK', 'US']
    
    for country in countries:
        # Target: new cases
        target_col = f"{country}_new_cases"
        if target_col in data.columns:
            target_columns.append(target_col)
        
        # Covariates
        cov_cols = [
            f"{country}_reproduction_rate",
            f"{country}_stringency_index",
            f"{country}_total_boosters_per_hundred"
        ]
        
        for col in cov_cols:
            if col in data.columns:
                covariate_columns.append(col)
    
    # Load and preprocess data
    processed_data = load_epidemic_data(
        data_path=data_path,
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        date_column='date',
        scaling_method='minmax'
    )
    
    return processed_data


def create_synthetic_data(
    n_regions: int = 6,
    n_timepoints: int = 200,
    n_covariates: int = 4,
    noise_level: float = 0.1,
    seasonal_period: int = 7,
    spatial_correlation: float = 0.3,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create synthetic epidemic data for testing.
    
    Args:
        n_regions: Number of regions
        n_timepoints: Number of time points
        n_covariates: Number of covariates
        noise_level: Level of noise to add
        seasonal_period: Seasonal period
        spatial_correlation: Spatial correlation strength
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with synthetic data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate time index
    dates = pd.date_range('2020-01-01', periods=n_timepoints, freq='D')
    
    # Generate covariates
    covariates = np.random.randn(n_regions, n_timepoints, n_covariates)
    
    # Generate targets with spatial correlation
    targets = np.zeros((n_regions, n_timepoints))
    
    for t in range(n_timepoints):
        # Base trend
        trend = 0.1 * t + 10 * np.sin(2 * np.pi * t / seasonal_period)
        
        # Regional variation
        regional_effect = np.random.randn(n_regions) * 2
        
        # Spatial correlation
        if t > 0:
            spatial_effect = spatial_correlation * np.mean(targets[:, t-1])
        else:
            spatial_effect = 0
        
        # Covariate effects
        covariate_effect = np.sum(covariates[:, t, :] * np.array([0.5, -0.3, 0.2, 0.1]), axis=1)
        
        # Combine effects
        targets[:, t] = trend + regional_effect + spatial_effect + covariate_effect + np.random.randn(n_regions) * noise_level
    
    # Ensure non-negative values
    targets = np.maximum(targets, 0)
    
    # Create region names
    regions = [f"Region_{i+1}" for i in range(n_regions)]
    
    return {
        'targets': targets,
        'covariates': covariates,
        'regions': regions,
        'dates': dates,
        'covariate_names': [f'Covariate_{i+1}' for i in range(n_covariates)]
    }


def load_owid_covid_data(
    data_path: str = "data/owid-covid-data.csv",
    countries: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load COVID-19 data from Our World in Data dataset.
    
    Args:
        data_path: Path to the OWID data file
        countries: List of countries to include
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Dictionary with processed OWID data
    """
    # Load raw data
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Filter by countries
    if countries is None:
        countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain']
    
    data = data[data['location'].isin(countries)]
    
    # Filter by date range
    if start_date:
        data = data[data['date'] >= start_date]
    if end_date:
        data = data[data['date'] <= end_date]
    
    # Define target and covariate columns
    target_columns = ['new_cases']
    covariate_columns = [
        'reproduction_rate',
        'stringency_index',
        'people_vaccinated_per_hundred',
        'icu_patients_per_million'
    ]
    
    # Pivot data to wide format
    target_data = data.pivot(index='date', columns='location', values='new_cases')
    target_data = target_data.fillna(0)
    
    # Process covariates
    covariate_data = {}
    for cov_col in covariate_columns:
        if cov_col in data.columns:
            cov_pivot = data.pivot(index='date', columns='location', values=cov_col)
            cov_pivot = cov_pivot.fillna(method='ffill').fillna(0)
            covariate_data[cov_col] = cov_pivot.values.T
    
    # Stack covariates
    if covariate_data:
        covariates = np.stack(list(covariate_data.values()), axis=-1)
    else:
        covariates = None
    
    # Get regions and dates
    regions = target_data.columns.tolist()
    dates = target_data.index.values
    
    return {
        'targets': target_data.values.T,
        'covariates': covariates,
        'regions': regions,
        'dates': dates,
        'covariate_names': list(covariate_data.keys())
    }


def create_spatial_matrix_from_coordinates(
    coordinates: np.ndarray,
    method: str = "distance",
    threshold: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Create spatial matrix from coordinates.
    
    Args:
        coordinates: Coordinate matrix (N x 2)
        method: Method for creating spatial weights
        threshold: Distance threshold
        **kwargs: Additional arguments
        
    Returns:
        Spatial matrix (N x N)
    """
    from ..utils.spatial import create_spatial_matrix
    
    return create_spatial_matrix(
        regions=list(range(len(coordinates))),
        method=method,
        threshold=threshold,
        coordinates=coordinates,
        **kwargs
    )


def create_treatment_indicator(
    regions: List[str],
    treated_regions: List[str],
    treatment_start_date: str,
    dates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create treatment indicator for causal inference.
    
    Args:
        regions: List of all regions
        treated_regions: List of treated regions
        treatment_start_date: Start date of treatment
        dates: Array of dates
        
    Returns:
        Tuple of (treatment_indicator, post_treatment_period)
    """
    # Treatment indicator (which regions are treated)
    treatment_indicator = np.array([1 if region in treated_regions else 0 for region in regions])
    
    # Post-treatment period indicator
    treatment_start = pd.Timestamp(treatment_start_date)
    post_treatment_period = np.array([1 if date >= treatment_start else 0 for date in dates])
    
    return treatment_indicator, post_treatment_period
