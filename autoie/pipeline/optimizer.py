from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa
from autoie.pipeline.base import Block
import six


from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))



def get_shape(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

@tf.function
def CRFloss(y_true, y_pred):
        # print("y_true.shape:", y_true.shape)
        # print("y_pred.shape:", y_pred.shape)


        # print("y_true:", y_true)
        # print("y_pred:", y_pred)
        #
        # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        batch_size, n_steps, _  = get_shape(y_pred)
        # print( batch_size, n_steps )
        y_true = tf.reshape(y_true, [batch_size, n_steps])
        y_true = tf.cast( y_true, dtype='int32' )
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(y_pred, y_true, [80] * 1024)
        loss = tf.reduce_mean(-log_likelihood)

        # log_likelihood, transition_params = tfa.text.crf_log_likelihood( y_pred, y_true, [80] * 128 )
        # loss = tf.keras.losses.sparse_categorical_crossentropy( y_true, y_pred )
        # loss = tf.reduce_sum(-log_likelihood)


        # loss = tf.reduce_sum( y_true - y_pred   )
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

