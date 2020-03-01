from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import shutil
import pandas as pd
import numpy as np
import inspect
import pkgutil
from collections import OrderedDict
import importlib
import tensorflow as tf
from tensorflow.python.util import nest


def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)


def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.mkdir(path)
    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        shutil.rmtree(path)
        os.mkdir(path)


def set_device(device_name):
    if device_name[0:3] == "cpu":
        cpus = tf.config.experimental.list_physical_devices('CPU')
        print("Available CPUs: {}".format(cpus))
        assert len(cpus) > 0, "Not enough CPU hardware devices available"
        cpu_idx = int(device_name[-1])
        tf.config.experimental.set_visible_devices(cpus[cpu_idx], 'CPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Available GPUs: {}".format(gpus))
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        gpu_idx = int(device_name[-1])
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')


# TODO: do we need this?
def load_dataframe_input(x):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        res = x
    elif isinstance(x, np.ndarray):
        res = pd.Series(x) if len(x.shape) == 1 else pd.DataFrame(x)
    elif isinstance(x, str):
        if not x.endswith('.csv'):
            raise TypeError(f'ONLY accept path to the local csv files')
        res = pd.read_csv(x)
    else:
        raise TypeError(f"cannot load {type(x)} into pandas dataframe")
    # make sure the returned dataframe's col name data type is string
    if isinstance(res, pd.DataFrame):
        res.columns = res.columns.astype('str')
    return res



def get_dicts(datas):
        w_all_dict, n_all_dict = {}, {}
        for sample in datas:
            for token, tag in zip(*sample):
                if token not in w_all_dict.keys():
                    w_all_dict[token] = 1
                else:
                    w_all_dict[token] += 1

                if tag not in n_all_dict.keys():
                    n_all_dict[tag] = 1
                else:
                    n_all_dict[tag] += 1

        sort_w_list = sorted(w_all_dict.items(), key=lambda d: d[1], reverse=True)
        sort_n_list = sorted(n_all_dict.items(), key=lambda d: d[1], reverse=True)
        w_keys = [x for x, _ in sort_w_list[:15999]]
        w_keys.insert(0, "UNK")

        n_keys = [x for x, _ in sort_n_list]
        w_dict = {x: i for i, x in enumerate(w_keys)}
        n_dict = {x: i for i, x in enumerate(n_keys)}
        return (w_dict, n_dict)

def w2num(datas, w_dict, n_dict):
        ret_datas = []
        for sample in datas:
            num_w_list, num_n_list = [], []
            for token, tag in zip(*sample):
                if token not in w_dict.keys():
                    token = "UNK"

                if tag not in n_dict:
                    tag = "O"

                num_w_list.append(w_dict[token])
                num_n_list.append(n_dict[tag])

            ret_datas.append((num_w_list, num_n_list, len(num_n_list)))
        return (ret_datas)

def len_norm(data_num, lens=80):
        ret_datas = []
        for sample1 in list(data_num):
            sample = list(sample1)
            ls = sample[-1]
            # print(sample)
            while (ls < lens):
                sample[0].append(0)
                ls = len(sample[0])
                sample[1].append(0)
            else:
                sample[0] = sample[0][:lens]
                sample[1] = sample[1][:lens]

            ret_datas.append(sample[:2])
        return (ret_datas)

