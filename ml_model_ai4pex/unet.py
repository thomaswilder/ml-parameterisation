#!/usr/bin/python3

"""U-Net model for spatial prediction tasks with configurable depth."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from ml_model_ai4pex.model_components import (
    ChannelSpatialSE, 
    ProvenanceMixin,
    ReplicationPadding2D, 
    Scenario
)


@keras.saving.register_keras_serializable(package="unet", name="UNet")
class UNet(ProvenanceMixin, keras.Model):
    '''
    A simple U-Net with configurable depth.

    Architecture (depth=2 example):
        Encoder:
            Block 1: 2x Conv(64)  -> MaxPool
            Block 2: 2x Conv(128) -> MaxPool
        Bottleneck:
            2x Conv(256)
        Decoder:
            UpSample + skip -> 2x Conv(128)
            UpSample + skip -> 2x Conv(64)
        Output:
            1x1 Conv -> n_targets
    '''
    def __init__(self, sc, input_shape,
                 dropout_rate=0.2, use_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.sc = sc
        self._input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        # --- Encoder ---
        self.encoder_blocks = []
        self.encoder_se = []
        self.pool_layers = []
        for i in range(sc.depth):
            filters = sc.base_filters * (2 ** i)
            block = self._make_conv_block(
                filters, use_dropout=(i < 2)
            )
            self.encoder_blocks.append(block)
            if use_attention:
                self.encoder_se.append(ChannelSpatialSE(filters))
            self.pool_layers.append(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # --- Bottleneck ---
        bottleneck_filters = sc.base_filters * (2 ** sc.depth)
        self.bottleneck = self._make_conv_block(bottleneck_filters)

        # --- Decoder ---
        self.upsample_layers = []
        self.decoder_blocks = []
        self.decoder_se = []
        for i in reversed(range(sc.depth)):
            filters = sc.base_filters * (2 ** i)
            self.upsample_layers.append(
                keras.layers.UpSampling2D(size=(2, 2))
            )
            self.decoder_blocks.append(
                self._make_conv_block(filters)
            )
            if use_attention:
                self.decoder_se.append(ChannelSpatialSE(filters))

        # --- Output ---
        n_targets = len(sc.target)
        self.output_conv = keras.layers.Conv2D(
            n_targets,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer=keras.initializers.GlorotUniform(),
            activation='linear',
        )

    def _make_conv_block(self, filters, use_dropout=False):
        '''Two conv layers with ReplicationPadding, BatchNorm, and ReLU.'''
        layers = [
            # first conv
            ReplicationPadding2D(padding=(1, 1)),
            keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(),
                activation=None,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # second conv
            ReplicationPadding2D(padding=(1, 1)),
            keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(),
                activation=None,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]
        if use_dropout:
            layers.append(keras.layers.SpatialDropout2D(self.dropout_rate))
        return keras.Sequential(layers)

    def call(self, inputs, training=False):
        # --- Encoder ---
        skips = []
        x = inputs
        print(f"Input shape: {x.shape}")
        
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            x = block(x, training=training)
            if self.use_attention:
                x = self.encoder_se[i](x)
            # print(f"Conv shape: {x.shape}")
            skips.append(x)          # save for skip connection
            x = pool(x)
            # print(f"Pool shape: {x.shape}")

        # --- Bottleneck ---
        x = self.bottleneck(x, training=training)
        # print(f"Bottleneck shape: {x.shape}")

        # --- Decoder ---
        for i, (upsample, block, skip) in enumerate(zip(
            self.upsample_layers,
            self.decoder_blocks,
            reversed(skips),
        )):
            x = upsample(x)
            # print(f"Upsample shape: {x.shape}")
            x = keras.layers.Concatenate()([x, skip])
            # print(f"Concat shape: {x.shape}")
            x = block(x, training=training)
            if self.use_attention:
                x = self.decoder_se[i](x)
            # print(f"Decoder Conv shape: {x.shape}")

        # --- Output ---
        y = self.output_conv(x)
        # print(f"Output shape: {y.shape}")
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'sc': self.sc,
            'input_shape': self._input_shape,
            'dropout_rate': self.dropout_rate,
            'use_attention': self.use_attention,
        })
        return config

    @classmethod
    def from_config(cls, config):
        sc_config = config.pop('sc')
        if isinstance(sc_config, dict) and 'config' in sc_config:
            sc_config = sc_config['config']
        sc = Scenario.from_config(sc_config)
        input_shape = config.pop('input_shape')
        return cls(sc, input_shape, **config)
