from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa
from autoie.pipeline.base import Block
from autoie.utils.common import get_shape


@tf.function
def CRFloss(y_true, y_pred):
        batch_size, n_steps, _  = get_shape(y_pred)
        y_true = tf.reshape(y_true, [batch_size, n_steps])
        y_true = tf.cast( y_true, dtype='int32' )
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(y_pred, y_true, [80] * batch_size)
        loss = tf.reduce_mean(-log_likelihood)
        return loss


class SparseCategoricalCrossentropyOptimizer(Block):
    """
    latent factor optimizer for cateory datas
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        output_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(9, activation='softmax')(output_node)
        return output_node

    @property
    def metric(self):
        return tf.keras.losses.sparse_categorical_crossentropy

    @property
    def loss(self):
        return tf.keras.losses.sparse_categorical_crossentropy


class CRFOptimizer(Block):
    """
    latent factor optimizer for cateory datas
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        output_node = tf.concat(inputs, axis=1)
        print( "crf output_node.shape:", output_node.shape )
        return output_node

    @property
    def metric(self):
        return CRFloss
        # return tf.keras.losses.sparse_categorical_crossentropy

    @property
    def loss(self):
        return CRFloss
        # return tf.keras.losses.sparse_categorical_crossentropy

