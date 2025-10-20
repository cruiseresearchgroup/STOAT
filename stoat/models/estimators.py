"""
Estimator implementations for STOAT.

This module contains the estimator classes that handle training and prediction
for the STOAT model, integrating the spatial causal inference and deep probabilistic forecasting.
"""

from functools import partial
from mxnet.gluon import HybridBlock
from gluonts.core.component import validated
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor
from gluonts.mx import (
    as_in_context,
    batchify,
    copy_parameters,
    get_hybrid_forward_input_names,
    GluonEstimator,
    RepresentableBlockPredictor,
    Trainer,
)
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    Transformation,
    InstanceSplitter,
    TestSplitSampler,
    SelectFields,
)

from .neural_networks import ProbabilisticTrainRNN, ProbabilisticPredRNN
from .distributions import get_distribution_output


class STOATEstimator(GluonEstimator):
    """
    STOAT estimator for spatial-temporal causal inference and probabilistic forecasting.
    
    This estimator integrates the spatial causal inference module with the deep
    probabilistic forecasting module to provide comprehensive epidemic forecasting.
    """
    
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        distr_output,
        num_cells: int,
        num_sample_paths: int = 100,
        scaling: bool = True,
        batch_size: int = 32,
        trainer: Trainer = Trainer(),
        spatial_matrix=None,
        causal_covariates=None,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling
        self.spatial_matrix = spatial_matrix
        self.causal_covariates = causal_covariates

    def create_transformation(self):
        """
        Create the transformation pipeline for the estimator.
        
        Returns:
            Transformation pipeline
        """
        return AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_training_data_loader(self, dataset, **kwargs):
        """
        Create the training data loader.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional arguments
            
        Returns:
            Training data loader
        """
        instance_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            ),
            past_length=self.context_length + 1,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_DYNAMIC_REAL,
                FieldName.OBSERVED_VALUES,
            ],
        )
        input_names = get_hybrid_forward_input_names(ProbabilisticTrainRNN)
        return TrainDataLoader(
            dataset=dataset,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
            **kwargs,
        )

    def create_training_network(self) -> ProbabilisticTrainRNN:
        """
        Create the training network.
        
        Returns:
            Training network instance
        """
        return ProbabilisticTrainRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        """
        Create the predictor for inference.
        
        Args:
            transformation: Data transformation pipeline
            trained_network: Trained network
            
        Returns:
            Predictor instance
        """
        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=self.context_length + 1,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_DYNAMIC_REAL,
                FieldName.OBSERVED_VALUES,
            ],
        )
        prediction_network = ProbabilisticPredRNN(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

        # Copy parameters from trained network
        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


def create_stoat_estimator(
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
    spatial_matrix=None,
    causal_covariates=None,
    **kwargs
) -> STOATEstimator:
    """
    Factory function to create a STOAT estimator with common configurations.
    
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
        spatial_matrix: Spatial relation matrix
        causal_covariates: Causal covariates
        **kwargs: Additional arguments
        
    Returns:
        Configured STOAT estimator
    """
    # Get distribution output
    distr_output = get_distribution_output(distribution)
    
    # Create trainer
    trainer = Trainer(
        ctx=ctx,
        epochs=epochs,
        learning_rate=learning_rate,
        hybridize=False,
        **kwargs
    )
    
    # Create estimator
    estimator = STOATEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
        distr_output=distr_output,
        num_cells=num_cells,
        num_sample_paths=num_sample_paths,
        scaling=scaling,
        batch_size=batch_size,
        trainer=trainer,
        spatial_matrix=spatial_matrix,
        causal_covariates=causal_covariates,
    )
    
    return estimator
