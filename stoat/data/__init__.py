"""
STOAT data processing modules.

This package contains data processing utilities for STOAT, including
data loaders, processors, and dataset preparation functions.
"""

from .processors import EpidemicDataProcessor
from .loaders import (
    load_epidemic_data,
    prepare_dataset,
    load_covid_data,
    create_synthetic_data,
    load_owid_covid_data,
    create_spatial_matrix_from_coordinates,
    create_treatment_indicator
)

__all__ = [
    'EpidemicDataProcessor',
    'load_epidemic_data',
    'prepare_dataset',
    'load_covid_data',
    'create_synthetic_data',
    'load_owid_covid_data',
    'create_spatial_matrix_from_coordinates',
    'create_treatment_indicator'
]
