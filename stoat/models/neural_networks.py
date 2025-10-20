"""
Neural network implementations for STOAT.

This module contains the core neural network architectures used in STOAT's
deep probabilistic forecasting module, including the probabilistic RNN implementations.
"""

import mxnet as mx
from mxnet import gluon
from gluonts.mx import MeanScaler, NOPScaler
from gluonts.mx import block
from gluonts.mx.block.dropout import VariationalZoneoutCell
from gluonts.core.component import validated
import gluonts


class ProbabilisticRNN(gluon.HybridBlock):
    """
    Base probabilistic RNN class for STOAT.
    
    This class implements the core RNN architecture used in STOAT's
    deep probabilistic forecasting module, supporting multiple LSTM layers
    with residual connections and variational dropout.
    """
    
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        distr_output,
        num_cells: int,
        num_sample_paths: int = 100,
        scaling: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.proj_distr_args = distr_output.get_args_proj()
        self.scaling = scaling

        with self.name_scope():
            # Build RNN architecture
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            
            # First LSTM layer
            cell = mx.gluon.rnn.LSTMCell(hidden_size=self.num_cells)
            self.rnn.add(cell)
            
            # Second LSTM layer with residual connection
            cell = mx.gluon.rnn.LSTMCell(hidden_size=self.num_cells)
            cell = mx.gluon.rnn.ResidualCell(cell)
            self.rnn.add(cell)
            
            # Third LSTM layer with variational dropout
            cell = mx.gluon.rnn.LSTMCell(hidden_size=self.num_cells)
            cell = VariationalZoneoutCell(
                base_cell=cell,
                zoneout_outputs=0.2,
                zoneout_states=0.1
            )
            self.rnn.add(cell)

            # Scaling
            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)
    
    def compute_scale(self, past_target, past_observed_values):
        """
        Compute scaling factor for the target values.
        
        Args:
            past_target: Past target values
            past_observed_values: Past observed values mask
            
        Returns:
            Scale factor
        """
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )
        return scale

    def unroll_encoder(
        self,
        F,
        past_target,
        past_observed_values,
        future_target=None,
        future_observed_values=None
    ):
        """
        Unroll the RNN encoder over the input sequence.
        
        Args:
            F: MXNet function interface
            past_target: Past target values
            past_observed_values: Past observed values mask
            future_target: Future target values (for training)
            future_observed_values: Future observed values mask (for training)
            
        Returns:
            Network output, states, and scale
        """
        if future_target is not None:  # Training mode
            target_in = F.concat(
                past_target, future_target, dim=-1
            ).slice_axis(
                axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
            )

            observed_values_in = F.concat(
                past_observed_values, future_observed_values, dim=-1
            ).slice_axis(
                axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
            )

            rnn_length = self.context_length + self.prediction_length
        else:  # Inference mode
            target_in = past_target.slice_axis(
                axis=1, begin=-(self.context_length + 1), end=-1
            )

            observed_values_in = past_observed_values.slice_axis(
                axis=1, begin=-(self.context_length + 1), end=-1
            )

            rnn_length = self.context_length

        # Compute scale
        scale = self.compute_scale(target_in, observed_values_in)

        # Scale target input
        target_in_scale = F.broadcast_div(target_in, scale)

        # Compute network output
        net_output, states = self.rnn.unroll(
            inputs=target_in_scale,
            length=rnn_length,
            layout="NTC",
            merge_outputs=True,
        )

        return net_output, states, scale


class ProbabilisticTrainRNN(ProbabilisticRNN):
    """
    Training version of the probabilistic RNN.
    
    This class implements the forward pass for training, computing the
    negative log-likelihood loss for the probabilistic forecasting task.
    """
    
    def hybrid_forward(
        self,
        F,
        past_target,
        future_target,
        past_observed_values,
        future_observed_values
    ):
        """
        Forward pass for training.
        
        Args:
            F: MXNet function interface
            past_target: Past target values
            future_target: Future target values
            past_observed_values: Past observed values mask
            future_observed_values: Future observed values mask
            
        Returns:
            Negative log-likelihood loss
        """
        net_output, _, scale = self.unroll_encoder(
            F, past_target, past_observed_values, future_target, future_observed_values
        )

        # Output target from -(context_length + prediction_length) to end
        target_out = F.concat(
            past_target, future_target, dim=-1
        ).slice_axis(
            axis=1, begin=-(self.context_length + self.prediction_length), end=None
        )

        # Project network output to distribution parameters
        distr_args = self.proj_distr_args(net_output)

        # Compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # Negative log-likelihood loss
        loss = distr.loss(target_out)
        return loss


class ProbabilisticPredRNN(ProbabilisticTrainRNN):
    """
    Prediction version of the probabilistic RNN.
    
    This class implements the forward pass for prediction, generating
    probabilistic forecasts through ancestral sampling.
    """
    
    def sample_decoder(self, F, past_target, states, scale):
        """
        Sample decoder for generating probabilistic forecasts.
        
        Args:
            F: MXNet function interface
            past_target: Past target values
            states: RNN states from encoder
            scale: Scaling factor
            
        Returns:
            Sampled forecasts
        """
        # Repeat states and scale for multiple sample paths
        repeated_states = [
            s.repeat(repeats=self.num_sample_paths, axis=0)
            for s in states
        ]
        repeated_scale = scale.repeat(repeats=self.num_sample_paths, axis=0)

        # First decoder input is the last value of past_target
        decoder_input = past_target.slice_axis(
            axis=1, begin=-1, end=None
        ).repeat(
            repeats=self.num_sample_paths, axis=0
        )

        # List to store samples at each time step
        future_samples = []

        # Generate samples for each future time step
        for k in range(self.prediction_length):
            rnn_outputs, repeated_states = self.rnn.unroll(
                inputs=decoder_input,
                length=1,
                begin_state=repeated_states,
                layout="NTC",
                merge_outputs=True,
            )

            # Project network output to distribution parameters
            distr_args = self.proj_distr_args(rnn_outputs)

            # Compute distribution
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # Draw samples
            new_samples = distr.sample()

            # Append samples for current time step
            future_samples.append(new_samples)

            # Update decoder input for next time step
            decoder_input = new_samples

        samples = F.concat(*future_samples, dim=1)

        # Reshape to (batch_size, num_samples, prediction_length)
        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))

    def hybrid_forward(self, F, past_target, past_observed_values):
        """
        Forward pass for prediction.
        
        Args:
            F: MXNet function interface
            past_target: Past target values
            past_observed_values: Past observed values mask
            
        Returns:
            Sampled forecasts
        """
        # Unroll encoder over context_length
        net_output, states, scale = self.unroll_encoder(
            F, past_target, past_observed_values
        )

        # Generate samples using decoder
        samples = self.sample_decoder(F, past_target, states, scale)

        return samples
