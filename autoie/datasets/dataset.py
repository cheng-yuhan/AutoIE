from abc import ABCMeta, abstractmethod

class Dataset(metaclass=ABCMeta):
    """AutoIE Dataset class.

    # Arguments
        data_path: String. The path to a directory where
            dataset is present. Defaults to None.
    """

    def __init__(self, data_path=None):
        self.train_file = data_path + 'train.txt'
        self.valid_file = data_path + 'valid.txt'
        self.test_file = data_path + 'test.txt'

    @abstractmethod
    def read(self):
        dataset = {}

        dataset['train'] = self.extract_tokens_and_tags(self.train_file)
        dataset['valid'] = self.extract_tokens_and_tags(self.valid_file)
        dataset['test'] = self.extract_tokens_and_tags(self.test_file)

        return dataset

    @abstractmethod
    def extract_tokens_and_tags(self, filename):
        """Extract tokens and tags from the given file.

        # Arguments
            filename: String. This is the file that has to
                be parsed to get the tokens and tags.
        """

        tokens_and_tags = []
        tokens = []
        tags = []

        try:
            with open(filename, encoding='utf-8') as handler:
                if handler.readline().rstrip().startswith('-DOCSTART-'):
                    for line in handler:
                        if not line.rstrip():
                            if len(tokens) > 0:
                                tokens_and_tags.append((tokens, tags))
                                tokens = []
                                tags = []
                        else:
                            token, _, _, tag = line.rstrip().split(' ')
                            tokens.append(token)
                            tags.append(tag)
                else:
                    handler.seek(0)

                    for line in handler:
                        if not line.rstrip():
                            if len(tokens) > 0:
                                tokens_and_tags.append((tokens, tags))
                                tokens = []
                                tags = []
                        else:
                            token, tag = line.rstrip().split('\t')
                            tokens.append(token)
                            tags.append(tag)
        except FileNotFoundError:
            print("File " + filename + " not found!")
            return None

        return tokens_and_tags
