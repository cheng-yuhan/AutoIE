# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
from autoie.datasets.dataloader import DataLoader
from autoie.utils.common import get_dicts, w2num, len_norm
from autoie.auto_search import Search
from autoie.tasks import NER
from autoie.pipeline import Input, OneHotEmbedding, BiLSTM, Dense, SparseCategoricalCrossentropyOptimizer


def ner_bilstm():
    # load dataset
    ds_rd = DataLoader("./examples/datasets/conll2003_v2/", "conll")
    dataset = ds_rd.read()
    w_dict, n_dict = get_dicts(dataset["train"])
    data_num = {}
    data_num["train"] = w2num(dataset["train"], w_dict, n_dict)
    data_norm = {}
    data_norm["train"] = len_norm(data_num["train"])
    train_data = np.array(data_norm["train"])
    train_x = train_data[:, 0, :]
    train_y = train_data[:, 1, :]
    print(train_x.shape)
    print(train_y.shape)

    # model
    input = Input(shape=(80))
    X = OneHotEmbedding()(input)
    X = BiLSTM()(X)
    X = BiLSTM()(X)
    X = Dense()(X)
    y = SparseCategoricalCrossentropyOptimizer()(X)
    model = NER(inputs=input, outputs=y)

    # search
    searcher = Search(model=model,
                      tuner='random',  # 'hyperband',
                      tuner_params={'max_trials': 100, 'overwrite': True},
                      )
    searcher.search(x=train_x, y=train_y, objective="val_sparse_categorical_crossentropy", batch_size=128,
                    validation_split=0.1)

if __name__ == "__main__":
    ner_bilstm()
