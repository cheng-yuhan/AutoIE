import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util import nest

from autoie.utils.common import dataset_shape
from autoie.pipeline import base


class Input(base.Node):
    """Input node for tensor data.
    The data should be numpy.ndarray or tf.data.Dataset.
    """

    def _check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to Input to be numpy.ndarray or '
                            'tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError('Expect the data to Input to be numerical, but got '
                            '{type}.'.format(type=x.dtype))

    def _convert_to_dataset(self, x):
        if isinstance(x, tf.data.Dataset):
            return x
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            return tf.data.Dataset.from_tensor_slices(x)

    def _record_dataset_shape(self, dataset):
        self.shape = dataset_shape(dataset)

    def fit_transform(self, x):
        dataset = self.transform(x)
        self._record_dataset_shape(dataset)
        return dataset

    def transform(self, x):
        """Transform x into a compatible type (tf.data.Dataset)."""
        self._check(x)
        dataset = self._convert_to_dataset(x)
        return dataset
