import json
import sys
from hazm import normalizer, word_tokenize
 
class DatasetItem: 
    def __init__(self, id, title, content, tags, date, url, category) -> None:
        self.id = id
        self.title = title
        self.content = content
        self.tags = tags
        self.date = date
        self.url = url
        self.category = category

    def __repr__(self) -> str:
        return str(self.__dict__)


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset_dict: dict = json.load(f)
    dataset: dict[str, DatasetItem] = {}
    for key in dataset_dict.keys():
        dataset[key] = DatasetItem(key, **dataset_dict[key])
    return dataset

def preprocess(dataset: dict[str, DatasetItem]):
    for k in dataset:
        print(dataset[k])

if __name__ == '__main__':
    if len(sys.argv) < 2 or not sys.argv[1]:
        sys.stderr.write('invalid usage: run ir-system.py my-dataset.json')
        exit(1)
    dataset = load_dataset(sys.argv[1])
    preprocess(dataset)
