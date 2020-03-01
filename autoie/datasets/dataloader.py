from autoie.datasets.conll import CoNLLDataset

class DataLoader:
    def __init__(self, data_path, name):
        if name == 'conll':
            self.dataset = CoNLLDataset(data_path)
            self.train_file = self.dataset.train_file
            self.valid_file = self.dataset.valid_file
            self.test_file = self.dataset.valid_file

    def read(self):
        return self.dataset.read()

    def extract_tokens_and_tags(self, filename):
        return self.dataset.extract_tokens_and_tags(filename)
