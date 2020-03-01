from abc import ABCMeta, abstractmethod

class Type1Dataset(metaclass=ABCMeta):
    def __init__(self, data_path=None):
        """Dataset Initialization"""
    
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
        """Method to extract tokens and tags."""
        tokens_and_tags = []
        tokens = []
        tags = []

        with open(filename, encoding='utf-8') as handler:
            for line in handler:
                if line.rstrip().startswith('-DOCSTART-'):
                    continue
                elif not line.rstrip():
                    if len(tokens) > 0:
                        tokens_and_tags.append((tokens, tags))
                        tokens = []
                        tags = []
                else:
                    token, _, _, tag = line.rstrip().split(' ')
                    tokens.append(token)
                    tags.append(tag)

        return tokens_and_tags


class Type2Dataset(metaclass=ABCMeta):
    def __init__(self, data_path=None):
        """Dataset Initialization"""
    
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
        """Method to extract tokens and tags."""
   
        tokens_and_tags = []
        tokens = []
        tags = []

        with open(filename, encoding='utf-8') as handler:
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

        return tokens_and_tags

