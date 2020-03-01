from tqdm import tqdm


class NerDatasetReader:
    def read(self, data_path):
        data_parts = ['train', 'valid', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = data_path + data_part + extension
            dataset[data_part] = self.read_file(str(file_path))
        return dataset

    def read_file(self, file_path):
        fileobj = open(file_path, 'r', encoding='utf-8')
        samples = []
        tokens = []
        tags = []

        for content in fileobj:

            content = content.strip('\n')

            if content == '-DOCSTART- -X- -X- O':
                pass
            elif content == '':
                if len(tokens) != 0:
                    samples.append((tokens, tags))
                    tokens = []
                    tags = []
            else:
                contents = content.split(' ')
                tokens.append(contents[0])
                tags.append(contents[-1])
        return samples
