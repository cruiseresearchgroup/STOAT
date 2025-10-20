"""
Distribution outputs for STOAT probabilistic forecasting.

This module provides various distribution outputs that can be used with the
deep probabilistic forecasting module, including Gaussian, Laplace, and Student's-t distributions.
"""

import mxnet as mx
from mxnet import gluon
from gluonts.mx.distribution import (
    GaussianOutput as GluonGaussianOutput,
    LaplaceOutput as GluonLaplaceOutput,
    StudentTOutput as GluonStudentTOutput,
)
from gluonts.core.component import validated


class GaussianOutput(GluonGaussianOutput):
    """
    Gaussian distribution output for probabilistic forecasting.
    
    This extends the GluonTS GaussianOutput with additional functionality
    specific to STOAT's spatial-temporal causal inference framework.
    """
    
    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def distribution(self, distr_args, scale=None, loc=None):
        """
        Create a Gaussian distribution from the distribution arguments.
        
        Args:
            distr_args: Distribution arguments (mean, std)
            scale: Scaling factor for the distribution
            loc: Location parameter
            
        Returns:
            Gaussian distribution object
        """
        return super().distribution(distr_args, scale=scale, loc=loc)


class LaplaceOutput(GluonLaplaceOutput):
    """
    Laplace distribution output for probabilistic forecasting.
    
    The Laplace distribution is particularly suitable for modeling
    epidemic data with heavy tails and potential outliers.
    """
    
    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def distribution(self, distr_args, scale=None, loc=None):
        """
        Create a Laplace distribution from the distribution arguments.
        
        Args:
            distr_args: Distribution arguments (mu, b)
            scale: Scaling factor for the distribution
            loc: Location parameter
            
        Returns:
            Laplace distribution object
        """
        return super().distribution(distr_args, scale=scale, loc=loc)


class StudentTOutput(GluonStudentTOutput):
    """
    Student's t-distribution output for probabilistic forecasting.
    
    The Student's t-distribution provides robust modeling of data
    with heavy tails and is particularly useful for epidemic forecasting
    where extreme values may occur.
    """
    
    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def distribution(self, distr_args, scale=None, loc=None):
        """
        Create a Student's t-distribution from the distribution arguments.
        
        Args:
            distr_args: Distribution arguments (nu, mu, sigma)
            scale: Scaling factor for the distribution
            loc: Location parameter
            
        Returns:
            Student's t-distribution object
        """
        return super().distribution(distr_args, scale=scale, loc=loc)


# Distribution registry for easy access
DISTRIBUTION_REGISTRY = {
    'gaussian': GaussianOutput,
    'laplace': LaplaceOutput,
    'student_t': StudentTOutput,
}


def get_distribution_output(distribution_name, **kwargs):
    """
    Get a distribution output by name.
    
    Args:
        distribution_name: Name of the distribution ('gaussian', 'laplace', 'student_t')
        **kwargs: Additional arguments for the distribution
        
    Returns:
        Distribution output instance
        
    Raises:
        ValueError: If distribution_name is not supported
    """
    if distribution_name not in DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unsupported distribution: {distribution_name}. "
            f"Supported distributions: {list(DISTRIBUTION_REGISTRY.keys())}"
        )
    
    return DISTRIBUTION_REGISTRY[distribution_name](**kwargs)
