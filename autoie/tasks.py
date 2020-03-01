from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autoie.pipeline.graph import HyperGraph
from tensorflow.python.util import nest
from autoie.pipeline import graph
from autoie.pipeline import base
from autoie.searcher.core import hyperparameters as hp_module
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NER(HyperGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

#
# class AutoModel(object):
#
#     def __init__(self,
#                  inputs,
#                  outputs,
#                  name='auto_model',
#                  max_trials=100,
#                  directory=None,
#                  objective='val_loss',
#                  tuner='greedy',
#                  overwrite=False,
#                  seed=None):
#         self.inputs = nest.flatten(inputs)
#         self.outputs = nest.flatten(outputs)
#         self.name = name
#         self.max_trials = max_trials
#         self.directory = directory
#         self.seed = seed
#         self.hyper_graph = None
#         self.objective = objective
#         # TODO: Support passing a tuner instance.
#
#         self.tuner = tuner_module.get_tuner_class(tuner)
#         self.overwrite = overwrite
#         self._split_dataset = False
#         if all([isinstance(output_node, base.Head)
#                 for output_node in self.outputs]):
#             self.heads = self.outputs
#         else:
#             self.heads = [output_node.in_blocks[0] for output_node in self.outputs]
#
#     def _meta_build(self, dataset):
#         # Using functional API.
#         self.hyper_graph = graph.HyperGraph(inputs=self.inputs,
#                                             outputs=self.outputs)
#
#
#         # if all([isinstance(output, base.Node) for output in self.outputs]):
#         #     self.hyper_graph = graph.HyperGraph(inputs=self.inputs,
#         #                                         outputs=self.outputs)
#         #
#         # # Using input/output API.
#         # elif all([isinstance(output, base.Head) for output in self.outputs]):
#         #     self.hyper_graph = meta_model.assemble(inputs=self.inputs,
#         #                                            outputs=self.outputs,
#         #                                            dataset=dataset,
#         #                                            seed=self.seed)
#         #     self.outputs = self.hyper_graph.outputs
#
#     def fit(self,
#             x=None,
#             y=None,
#             epochs=None,
#             callbacks=None,
#             validation_split=0.2,
#             validation_data=None,
#             **kwargs):
#         dataset, validation_data = self._prepare_data(
#             x=x,
#             y=y,
#             validation_data=validation_data,
#             validation_split=validation_split)
#
#         # Initialize the hyper_graph.
#         self._meta_build(dataset)
#
#
#
#         # Initialize the Tuner.
#         # The hypermodel needs input_shape, which can only be known after
#         # preprocessing. So we preprocess the dataset once to get the input_shape,
#         # so that the hypermodel can be built in the initializer of the Tuner, which
#         # does not access the dataset.
#         hp = hp_module.HyperParameters()
#
#         preprocess_graph, keras_graph = self.hyper_graph.build_graphs(hp)
#
#         preprocess_graph.preprocess(
#             dataset=dataset,
#             validation_data=validation_data,
#             fit=True)
#
#         self.tuner = self.tuner(
#             hyper_graph=self.hyper_graph,
#             hypermodel=keras_graph,
#             fit_on_val_data=self._split_dataset,
#             overwrite=self.overwrite,
#             objective=self.objective,
#             max_trials=self.max_trials,
#             directory=self.directory,
#             seed=self.seed,
#             project_name=self.name)
#
#         # Process the args.
#         if callbacks is None:
#             callbacks = []
#         if epochs is None:
#             epochs = 1000
#             if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
#                         for callback in callbacks]):
#                 callbacks = callbacks + [
#                     tf.keras.callbacks.EarlyStopping(patience=10)]
#
#         self.tuner.search(x=dataset,
#                           epochs=epochs,
#                           callbacks=callbacks,
#                           validation_data=validation_data,
#                           **kwargs)
#
#     def _process_xy(self, x, y=None, fit=False, predict=False):
#         """Convert x, y to tf.data.Dataset.
#         # Arguments
#             x: Any type allowed by the corresponding input node.
#             y: Any type allowed by the corresponding head.
#             fit: Boolean. Whether to fit the type converter with the provided data.
#             predict: Boolean. If it is called by the predict function of AutoModel.
#         # Returns
#             A tf.data.Dataset containing both x and y.
#         """
#         if isinstance(x, tf.data.Dataset):
#             if y is None and not predict:
#                 return x
#             if isinstance(y, tf.data.Dataset):
#                 return tf.data.Dataset.zip((x, y))
#
#         x = nest.flatten(x)
#         new_x = []
#         for data, input_node in zip(x, self.inputs):
#             if fit:
#                 data = input_node.fit_transform(data)
#             else:
#                 data = input_node.transform(data)
#             new_x.append(data)
#         x = tf.data.Dataset.zip(tuple(new_x))
#
#         if predict:
#             return tf.data.Dataset.zip((x, x))
#
#         if not isinstance(y, tf.data.Dataset):
#             y = nest.flatten(y)
#             new_y = []
#             for data, head_block in zip(y, self.heads):
#                 if fit:
#                     data = head_block.fit_transform(data)
#                 else:
#                     data = head_block.transform(data)
#                 new_y.append(data)
#             y = tf.data.Dataset.zip(tuple(new_y))
#
#         return tf.data.Dataset.zip((x, y))
#
#     def _prepare_data(self, x, y, validation_data, validation_split):
#         """Convert the data to tf.data.Dataset."""
#         # Check validation information.
#         if not validation_data and not validation_split:
#             raise ValueError('Either validation_data or validation_split '
#                              'should be provided.')
#         # TODO: Handle other types of input, zip dataset, tensor, dict.
#         # Prepare the dataset.
#         dataset = self._process_xy(x, y, fit=True)
#         if validation_data:
#             self._split_dataset = False
#             val_x, val_y = validation_data
#             validation_data = self._process_xy(val_x, val_y)
#         # Split the data with validation_split.
#         if validation_data is None and validation_split:
#             self._split_dataset = True
#             dataset, validation_data = utils.split_dataset(dataset, validation_split)
#         return dataset, validation_data
#
#     def predict(self, x, batch_size=32, **kwargs):
#         """Predict the output for a given testing data.
#         # Arguments
#             x: Any allowed types according to the input node. Testing data.
#             batch_size: Int. Defaults to 32.
#             **kwargs: Any arguments supported by keras.Model.predict.
#         # Returns
#             A list of numpy.ndarray objects or a single numpy.ndarray.
#             The predicted results.
#         """
#         preprocess_graph, model = self.tuner.get_best_model()
#         x = preprocess_graph.preprocess(
#             self._process_xy(x, None, predict=True))[0].batch(batch_size)
#         y = model.predict(x, **kwargs)
#         y = self._postprocess(y)
#         if isinstance(y, list) and len(y) == 1:
#             y = y[0]
#         return y
#
#     def _postprocess(self, y):
#         y = nest.flatten(y)
#         new_y = []
#         for temp_y, head_block in zip(y, self.heads):
#             if isinstance(head_block, base.Head):
#                 temp_y = head_block.postprocess(temp_y)
#             new_y.append(temp_y)
#         return new_y
#
#     def evaluate(self, x, y=None, batch_size=32, **kwargs):
#         """Evaluate the best model for the given data.
#         # Arguments
#             x: Any allowed types according to the input node. Testing data.
#             y: Any allowed types according to the head. Testing targets.
#                 Defaults to None.
#             batch_size: Int. Defaults to 32.
#             **kwargs: Any arguments supported by keras.Model.evaluate.
#         # Returns
#             Scalar test loss (if the model has a single output and no metrics) or
#             list of scalars (if the model has multiple outputs and/or metrics).
#             The attribute model.metrics_names will give you the display labels for
#             the scalar outputs.
#         """
#         preprocess_graph, model = self.tuner.get_best_model()
#         data = preprocess_graph.preprocess(
#             self._process_xy(x, y))[0].batch(batch_size)
#         return model.evaluate(data, **kwargs)
#

