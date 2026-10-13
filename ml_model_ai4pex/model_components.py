#!/usr/bin/python3
"""Shared model components including custom Keras layers and the Scenario dataclass."""

import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras


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
