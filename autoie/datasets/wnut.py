from autoie.datasets.dataloader import Type2Dataset

class WNUTDataset(Type2Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)

    def read(self):
        return super().read()

    def extract_tokens_and_tags(self, filename):
        return super().extract_tokens_and_tags(filename)
