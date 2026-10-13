#!/usr/bin/python3
"""Convolutional Neural Network (CNN) model implementation using Keras API."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

from ml_model_ai4pex.model_components import (
    ProvenanceMixin,
    ReplicationPadding2D,
    Scenario,
)

@keras.saving.register_keras_serializable(package="cnn", name="CNN")
class CNN(ProvenanceMixin, keras.Model):
    '''
    A fully convolutional neural network class using keras api
    '''
    def __init__(self, sc, 
                 input_shape, 
                 dropout_rate=0.2,
                 **kwargs
                 ):
        super().__init__(**kwargs) # inherit from keras.Model class
        self.sc = sc
        self.input_shape = input_shape #// this doesn't look it is used?
        self.dropout_rate = dropout_rate

        # //self.input_layer = ReplicationPadding2D(padding=(2, 2), input_shape=input_shape)

        self.conv_blocks = []
        for i, (f, k, p, d) in \
                enumerate(zip(sc.filters[:-1], sc.kernels[:-1], sc.padding[:-1], sc.dilation_rates[:-1])):
            if i<2:
                use_dropout = True
            block = self._make_conv_block(f, k, p, d, use_dropout=use_dropout)
            self.conv_blocks.append(block)
            use_dropout = False # reset to false to ensure only first two blocks have dropout
            # self.conv_blocks = [self._make_conv_block(f, k, p) for f, k, p in 
            #                 zip(sc.filters[:-1], sc.kernels[:-1], sc.padding[:-1])]

        self.output_layer = (keras.layers.Conv2D(
                        sc.filters[-1], 
                        sc.kernels[-1], 
                        use_bias=False,
                        kernel_initializer=keras.initializers.GlorotUniform(),
                        activation='linear')
                        )

    def _make_conv_block(self, filters, kernel, padding, dilation_rate=1, use_dropout=False):
        layers = [
            ReplicationPadding2D(padding=padding),
            keras.layers.Conv2D(
                filters, 
                kernel_size=kernel,
                dilation_rate=dilation_rate,
                use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(),
                activation="relu"), 
            keras.layers.BatchNormalization()
        ]
        if use_dropout:
            # layers.append(keras.layers.Dropout(self.dropout_rate))
            layers.append(keras.layers.SpatialDropout2D(self.dropout_rate))
        return keras.Sequential(layers)

    # def get_padded_feature_map(self, inputs):
    #     '''
    #         Returns a feature map with 2d replication padding applied.
    #     '''
    #     x = inputs
    #     x = ReplicationPadding2D(padding=(2, 2))(x) 
    #     return x.numpy()  # Returns as numpy array

    #/ custom objects require the below to be serializable
    def get_config(self):
        config = super().get_config()
        config.update({
            'sc': self.sc,
            'input_shape': self.input_shape,
            'dropout_rate': self.dropout_rate,
        })
        return config

    #/ then must deserialize the custom object
    @classmethod
    def from_config(cls, config):
        sc_config = config.pop('sc')
        # extracts the inner config from the Scenario object
        if isinstance(sc_config, dict) and 'config' in sc_config:
            sc_config = sc_config['config']
        sc = Scenario.from_config(sc_config) # reconstruct Scenario object
        input_shape = config.pop('input_shape')
        dropout_rate = config.pop('dropout_rate', 0.2)
        # models saved while the squeeze-excitation option existed carry
        # `use_attention` in their config: false is dropped, true cannot be rebuilt
        if config.pop('use_attention', False):
            raise ValueError(
                "This model was saved with use_attention=True. The "
                "squeeze-excitation layers it needs were removed from "
                "ml_model_ai4pex, so it can no longer be rebuilt."
            )
        return cls(sc, 
                   input_shape, 
                   dropout_rate=dropout_rate, 
                   **config
        )

    def call(self, inputs, training=False):
        x = inputs
        # //x = self.input_layer(inputs) # adding replication padding
        for block in self.conv_blocks:
            x = block(x, training = training) # training flag is passed depending on .fit or .evaluate/.predict
        x = ReplicationPadding2D(padding=self.sc.padding[-1])(x) # add padding before final convolution
        y = self.output_layer(x) # final convolution layer with padding
        #// y = keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(yp) # crop the padding in final layer
        return y