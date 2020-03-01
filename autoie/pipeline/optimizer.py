from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autoie.pipeline.base import Block


class SparseCategoricalCrossentropyOptimizer(Block):
    """
    latent factor optimizer for cateory datas
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        output_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(9, activation='softmax')(output_node)
        # output_node = tf.reshape(output_node, [1000,80])
        # output_node = tf.reshape(output_node, [-1])
        # output_node = tf.keras.layers.Flatten()( output_node )
        print("output_node shape", output_node)
        return output_node

    @property
    def metric(self):
        # return tf.keras.metrics.MeanSquaredError(name='mse')
        # return tf.keras.losses.categorical_crossentropy
        return tf.keras.losses.sparse_categorical_crossentropy

    @property
    def loss(self):
        # return tf.keras.losses.MeanSquaredError(name='mse')
        # return tf.keras.losses.categorical_crossentropy
        return tf.keras.losses.sparse_categorical_crossentropy
