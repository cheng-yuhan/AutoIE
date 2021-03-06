# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from autoie.datasets.dataloader import DataLoader
from autoie.auto_search import Search
from autoie.tasks import NER
from autoie.pipeline import Input, OneHotEmbedding, BiLSTM, Dense, SparseCategoricalCrossentropyOptimizer

# load dataset
ds_rd = DataLoader("./examples/datasets/conll2003_v2/", "conll")
train_x, val_x, train_y, val_y = ds_rd.read()
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

# model
input = Input(shape=(80))
output = OneHotEmbedding()(input)
output = BiLSTM()(output)
output = BiLSTM()(output)
output = Dense()(output)
output = SparseCategoricalCrossentropyOptimizer()(output)
model = NER(inputs=input, outputs=output)

# search
searcher = Search(model=model,
                  tuner='random',  # 'hyperband',
                  tuner_params={'max_trials': 2, 'overwrite': True},
                  )
searcher.search(x=train_x, y=train_y, x_val=val_x, y_val=val_y, objective="val_sparse_categorical_crossentropy",
                batch_size=2048, epochs = 10)
