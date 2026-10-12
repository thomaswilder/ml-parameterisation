#!/usr/bin/python3
"""Shared model components including custom Keras layers and the Scenario dataclass."""

import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops


@dataclass
@keras.saving.register_keras_serializable(package="cnn", name="Scenario")
class Scenario:
    '''
    Scenario dataclass. Can easily setup the design,
        adjusting input variables, target variables, cnn filters and kernels.
    '''
    input_var: Iterable[str]
    target: Iterable[str]
    filters: List[int] = None
    kernels: Tuple[int, int] = None
    padding: Tuple[int, int] = None
    dilation_rates: List[int] = None
    name: str = None
    base_filters: int = None
    depth: int = None

    def __post_init__(self):
        if self.filters is not None and self.kernels is not None:
            self._validate_kernel_padding()

    def _validate_kernel_padding(self):
        for idx, (kernel, pad, dilation) in \
                enumerate(zip(self.kernels, self.padding, self.dilation_rates)):
            kernel = tuple(kernel) if isinstance(kernel, list) else kernel
            pad = tuple(pad) if isinstance(pad, list) else pad

            expected_pad = (
                dilation * (kernel[0] - 1) // 2,
                dilation * (kernel[1] - 1) // 2,
            )
            if pad != expected_pad:
                raise ValueError(
                    f"Padding at index {idx} {pad} "
                    f"does not match expected {expected_pad} "
                    f"for kernel {kernel}"
                )

    def get_config(self):
        return {
            'input_var': list(self.input_var),
            'target': list(self.target),
            'filters': self.filters,
            'kernels': self.kernels,
            'padding': self.padding,
            'dilation_rates': self.dilation_rates,
            'name': self.name,
            'base_filters': self.base_filters,
            'depth': self.depth,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="cnn", name="ReplicationPadding2D")
class ReplicationPadding2D(keras.layers.Layer):
    '''
        2D Replication padding

        Attributes:

            - padding : (padding_width, padding_height) tuple

        From:
            https://github.com/christianversloot/machine-learning-articles/blob/main/using-constant-padding-reflection-padding-and-replication-padding-with-keras.md
    '''
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(
            input_tensor,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
            'SYMMETRIC',
        )


@keras.saving.register_keras_serializable(package="cnn", name="MSESSIMLoss")
class MSESSIMLoss(keras.losses.Loss):
    '''
    Custom loss function that combines MSE and SSIM losses.
    '''
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        super(MSESSIMLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        mse_loss = ops.mean(ops.square(y_true - y_pred))
        max_value = ops.abs(tf.reduce_max(y_true) - tf.reduce_min(y_true))
        ssim_loss = 1 - tf.image.ssim(
            y_true,
            y_pred,
            max_val=max_value,
            filter_size=11,
        )
        return self.alpha * mse_loss + (1 - self.beta) * ssim_loss


@keras.saving.register_keras_serializable(package="cnn", name="MaskedMSELoss")
class MaskedMSELoss(keras.losses.Loss):
    '''
    Custom loss function that masks the input so land values
        do not contribute to the loss.
    '''
    def __init__(self, mask=None, **kwargs):
        super(MaskedMSELoss, self).__init__(**kwargs)

        if mask is not None:
            self.mask = tf.constant(mask, dtype=tf.float32)
        else:
            self.mask = None

    def call(self, y_true, y_pred):
        y_true = y_true * self.mask if self.mask is not None else y_true
        y_pred = y_pred * self.mask if self.mask is not None else y_pred

        if self.mask is not None:
            mse = ops.sum(ops.square(y_true - y_pred)) / ops.sum(self.mask)
        else:
            mse = ops.mean(ops.square(y_true - y_pred))

        return mse


@keras.saving.register_keras_serializable(package="cnn", name="ChannelSpatialSE")
class ChannelSpatialSE(keras.layers.Layer):
    '''
    Channel Squeeze-Excitation block.
    Learns to reweight feature channels based on global context.

    References:
        - Squeeze and Excitation Networks (https://arxiv.org/abs/1709.01507)
        - Also see https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se.py
        - See Roy et al for combined spatial and channel SE: https://arxiv.org/pdf/1803.02579
    '''
    def __init__(self, filters, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.gap = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = keras.layers.Dense(
            filters // ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False,
        )
        self.fc2 = keras.layers.Dense(
            filters,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
        )
        self.spatial_conv = keras.layers.Conv2D(
            1,
            (1, 1),
            activation='sigmoid',
            use_bias=False,
            kernel_initializer='he_normal',
        )

    def call(self, x):
        cw = self.gap(x)
        cw = self.fc1(cw)
        cw = self.fc2(cw)
        cw = x * cw
        sw = self.spatial_conv(x)
        sw = x * sw
        return cw + sw

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config


def _to_json(obj):
    '''Fallback for json.dump: numpy scalars/arrays -> python, else str.'''
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


class ProvenanceMixin:
    '''
    Stores a training record (a JSON-serialisable dict in `self.provenance`)
    inside the .keras archive as assets/provenance.json, and restores it
    on keras.saving.load_model. Keras calls save_assets/load_assets on the
    model during save/load.

    Models saved before this was added load with provenance = None.
    Read without loading the model:
        unzip -p model.keras assets/provenance.json
    '''
    provenance = None
    _provenance_filename = 'provenance.json'

    def save_assets(self, dir_path):
        if self.provenance is None:
            return
        with open(os.path.join(dir_path, self._provenance_filename), 'w') as f:
            json.dump(self.provenance, f, indent=2, default=_to_json)

    def load_assets(self, dir_path):
        path = os.path.join(dir_path, self._provenance_filename)
        if os.path.exists(path):
            with open(path) as f:
                self.provenance = json.load(f)
