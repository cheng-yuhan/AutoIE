from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.util import nest
from autoie.pipeline.base import Block
from tensorflow.keras.layers import Dense, Input, Concatenate
import random
import tensorflow as tf
from autoie.pipeline.base import Block
from tensorflow.keras.layers import LSTM, Bidirectional


class OneHotEmbedding(Block):
    """
    latent factor mapper for single categorical feature
    """

    def __init__(self,
                 id_num=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.id_num = id_num
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update({
            'id_num': self.id_num,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.id_num = state['id_num']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        # input_node = inputs
        print("inputs", inputs)
        input_node = tf.concat(inputs, axis=1)
        id_num = self.id_num or hp.Choice('id_num', [16000], default=16000)
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16, 32, 64, 256], default=32)
        output_node = tf.keras.layers.Embedding(id_num, embedding_dim, input_length=80)(input_node)
        return output_node


class BiLSTM(Block):
    """
    multi-layer perceptron interactor
    """

    def __init__(self,
                 units=None,
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
                 activation=None,
                 merge_mode=None,
                 use_bias=None,
                 recurrent_dropout_rate=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.activation = activation
        self.merge_mode = merge_mode

    def get_state(self):
        state = super().get_state()
        state.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.units = state['units']
        self.num_layers = state['num_layers']
        self.use_batchnorm = state['use_batchnorm']
        self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        input_node = tf.concat(inputs, axis=1)
        output_node = input_node

        units = self.units or hp.Choice('units', [64, 256, 512, 1024], default=256)
        activation = self.activation or hp.Choice('activation', ["tanh", "relu", "sigmoid", "selu", "elu"],
                                                  default="tanh")
        merge_mode = self.merge_mode or hp.Choice("merge_mode", ['sum', 'mul', 'concat', 'ave'], default="concat")
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)
        recurrent_dropout_rate = self.recurrent_dropout_rate or hp.Choice('recurrent_dropout_rate',
                                                                          [0.0, 0.25, 0.5],
                                                                          default=0)

        output_node = Bidirectional(
            LSTM(units=units, activation=activation, return_sequences=True, dropout=dropout_rate,
                 recurrent_dropout=recurrent_dropout_rate),
            merge_mode=merge_mode)(output_node)
        return output_node


class Dense(Block):
    """
    multi-layer perceptron interactor
    """

    def __init__(self,
                 units=None,
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = []
        self.tunable_candidates = ['units', 'num_layers', 'use_batchnorm', 'dropout_rate']
        self.units = units
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        # self._check_fixed()
        # self.hyperparameters = self._get_hyperparameters()

    def get_state(self):
        state = super().get_state()
        state.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.units = state['units']
        self.num_layers = state['num_layers']
        self.use_batchnorm = state['use_batchnorm']
        self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        input_node = tf.concat(inputs, axis=1)
        output_node = input_node

        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Choice('use_batchnorm', [True, False], default=False)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)

        for i in range(num_layers):
            units = self.units or hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)

            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node



#
#
# class HyperInteraction(Block):
#     """Combination of serveral interactor into one.
#     # Arguments
#     meta_interator_num: int
#     interactor_type: interactor_name
#     """
#     def __init__(self, meta_interator_num=None, interactor_type=None, **kwargs):
#         super().__init__(**kwargs)
#         self.meta_interator_num = meta_interator_num
#         self.interactor_type = interactor_type
#
#     def get_state(self):
#         state = super().get_state()
#         state.update({
#             'interactor_type': self.interactor_type,
#             'meta_interator_num': self.meta_interator_num
#         })
#         return state
#
#     def set_state(self, state):
#         super().set_state(state)
#         self.interactor_type = state['interactor_type']
#         self.meta_interator_num = state['meta_interator_num']
#
#     def build(self, hp, inputs=None):
#         inputs = nest.flatten(inputs)
#         meta_interator_num =  self.meta_interator_num or hp.Choice('meta_interator_num',
#                                                                     [1, 2, 3, 4, 5, 6],
#                                                                     default=3)
#         # inputs = tf.keras.backend.repeat(inputs, n=meta_interator_num)
#         # interactors_name = ["MLPInteraction"]
#         interactors_name = []
#         for i in range( meta_interator_num ):
#             tmp_interactor_type = self.interactor_type or hp.Choice('interactor_type_' + str(i),
#                                                                     [ "MLPInteraction", "MLPInteraction", "MLPInteraction"],
#                                                                     default='MLPInteraction')
#             interactors_name.append(tmp_interactor_type)
#
#
#         print( "interactors_name", interactors_name )
#         outputs = []
#         for i, interactor_name in enumerate( interactors_name ):
#             if interactor_name == "MLPInteraction":
#                 ##TODO: support intra block hyperparameter tuning
#                 output_node = MLPInteraction().build(hp, inputs)
#                 outputs.append(output_node)
#
#             if interactor_name == "ConcatenateInteraction":
#                 output_node = MLPInteraction().build(hp, inputs)
#                 outputs.append(output_node)
#
#             if interactor_name == "RandomSelectInteraction":
#                 output_node = MLPInteraction().build(hp, inputs)
#                 outputs.append(output_node)
#
#         outputs = tf.concat(outputs, axis=1)
#         # ConcatenateInteraction().build(hp, inputs)
#         return outputs
#
