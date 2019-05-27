import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import backend as K

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import nn


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization layer (Ba et al., 2016).
    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.
    Arguments:
    axis: Integer or List/Tuple. The axis that should be normalized
      (typically the features axis).
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    trainable: Boolean, if `True` the variables will be marked as trainable.
    Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
    Output shape:
    Same shape as input.
    References:
    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    """

    def __init__(self,
               axis=-1,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(LayerNormalization, self).__init__(
            name=name, trainable=trainable, **kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise ValueError('Expected an int or a list/tuple of ints for the argument \'axis\', but received instead: %s' % axis)

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError('Input shape %s has undefined rank.' % input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(
              name='gamma',
              shape=param_shape,
              initializer=self.gamma_initializer,
              regularizer=self.gamma_regularizer,
              constraint=self.gamma_constraint,
              trainable=True)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
              name='beta',
              shape=param_shape,
              initializer=self.beta_initializer,
              regularizer=self.beta_regularizer,
              constraint=self.beta_constraint,
              trainable=True)
        else:
            self.beta = None

    def call(self, inputs):
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)

        # Calculate the moments on the last axis (layer activations).
        mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

        # Broadcasting only necessary for norm where the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value
        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and self.axis != [ndims - 1]):
                return array_ops.reshape(v, broadcast_shape)
            return v
        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        # Compute layer normalization using the batch_normalization function.
        outputs = nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=offset,
            scale=scale,
            variance_epsilon=self.epsilon)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

