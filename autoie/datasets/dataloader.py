import numpy as np
from autoie.datasets.conll import CoNLLDataset
from autoie.datasets.wnut import WNUTDataset
from autoie.datasets.re3d import Re3dDataset
from autoie.datasets.gum import GUMDataset
from autoie.datasets.sec import SECDataset
from autoie.utils.common import get_dicts, w2num, len_norm

DATASET_CLASSES = {
    'conll': CoNLLDataset,
    'wnut': WNUTDataset,
    're3d': Re3dDataset,
    'gum': GUMDataset,
    'sec': SECDataset,
}

class DataLoader:
    """AutoIE DataLoader class.

    # Arguments
        data_path: String. The path to a directory where
            dataset is present. Defaults to None.
        name: String. This gives the name of the dataset.
            Defaults to None. "name" should be one of
            "conll", "wnut", "re3d", "gum" or "sec"
    """

    def __init__(self, data_path, name):
        if isinstance(name, str) and name in DATASET_CLASSES:
            dataset_class = DATASET_CLASSES.get(name)
            self.dataset = dataset_class(data_path)

            self.train_file = self.dataset.train_file
            self.valid_file = self.dataset.valid_file
            self.test_file = self.dataset.test_file
        else:
            raise ValueError('The value {name} passed for argument name is invalid, '
                             'expected one of "conll", "wnut", "re3d", '
                             '"gum", "sec".'.format(name=name))

    def read(self):
        dataset = self.dataset.read()
        w_dict, n_dict = get_dicts(dataset["train"])

        data_num = {}
        data_num["train"] = w2num(dataset["train"], w_dict, n_dict)

        data_norm = {}
        data_norm["train"] = len_norm(data_num["train"])
        train_data = np.array(data_norm["train"])

        return train_data[:, 0, :], train_data[:, 1, :]

    def extract_tokens_and_tags(self, filename):
        """Extract tokens and tags from the given file.

        # Arguments
            filename: String. This is the file that has to
                be parsed to get the tokens and tags.
        """

        return self.dataset.extract_tokens_and_tags(filename)
