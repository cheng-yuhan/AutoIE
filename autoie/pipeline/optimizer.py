from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tf_ad
from autoie.pipeline.base import Block


from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# class CRFloss(Loss):
#   def call(self, y_true, y_pred):
#     y_pred = ops.convert_to_tensor(y_pred)
#     y_true = math_ops.cast(y_true, y_pred.dtype)
#
#     output_node, self.transition_params = tf_ad.text.crf_log_likelihood(y_pred, y_true, 80)
#     loss = tf.keras.losses.sparse_categorical_crossentropy( y_pred, y_true )
#     return loss
@tf.function
def CRFloss(_y_true, _y_pred):
    # y_pred = ops.convert_to_tensor(y_pred)
    # y_true = math_ops.cast(y_true, y_pred.dtype)
    # with tf.init_scope():

        # _y_true = tf.convert_to_tensor(_y_true, dtype=tf.int64)
        # print( "_y_true.shape:", _y_true.shape )
        # print("_y_pred.shape:", _y_pred.shape)
        log_likelihood, transition_params = tf_ad.text.crf_log_likelihood(_y_pred, _y_true, 80)
        # loss = tf.keras.losses.sparse_categorical_crossentropy( y_pred, y_true )
        loss = tf.reduce_sum(-log_likelihood)


        # loss = tf.reduce_sum( _y_true - _y_pred   )
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
        # return tf.keras.losses.sparse_categorical_crossentropy
        return CRFloss
