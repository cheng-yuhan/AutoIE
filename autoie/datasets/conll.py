from autoie.datasets.dataset import Type1Dataset
from nltk.corpus import conll2002

class CoNLLDataset(Type1Dataset):
    def __init__(self, data_path=None):
        if data_path is None:
            self.train_file = None
            return

        super().__init__(data_path)

    def read(self):
        return super().read()

    def extract_tokens_and_tags(self, filename):
        tokens_and_tags = []

        if filename is None:
            train_sents = list(conll2002.iob_sents(u'esp.train'))
            for train_sent in train_sents:
                tokens_and_tags.append(([x for x, _, _ in train_sent], [x for _, _, x in train_sent]))
            return tokens_and_tags

        return super().extract_tokens_and_tags(filename)
