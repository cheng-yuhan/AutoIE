from autoie.datasets.dataset import Dataset

class GUMDataset(Dataset):
    """Dataset class for GUM.

    # Arguments
        data_path: String. The path to a directory where
            dataset is present. Defaults to None.
    """

    def __init__(self, data_path):
        super().__init__(data_path)

    def read(self):
        return super().read()

    def extract_tokens_and_tags(self, filename):
        """Extract tokens and tags from the given file.

        # Arguments
            filename: String. This is the file that has to
                be parsed to get the tokens and tags.
        """

        return super().extract_tokens_and_tags(filename)
