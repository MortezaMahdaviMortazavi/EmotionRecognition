import torch
import numpy as np
import config

from preprocessing import Preprocessing
from vocabulary import Vocabulary

class ArmanDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_type):
        assert dataset_type in ['train','val']
        self.dataset_type = dataset_type # train or val
        self.texts = [] # the text in each sample of dataset
        self.targets = [] # the labels or targets of each sample
        self.tokenizer = config.TOKENIZER # tokenizer that tokenize the text
        self.preprocessor = Preprocessing(dataset='arman') # return text and its target while we call it with a text input
        self.has_target = isinstance(self.targets, list) or isinstance(self.targets, np.ndarray)


        self.labels_dict = self.preprocessor.labels
        self.vocab = Vocabulary(self.texts,vocab_threshold=2,name='arman',load=True)
        self.extract_data()
    
    def extract_data(self):
        file_path = config.ARMAN_TRAIN if self.dataset_type == 'train' else config.ARMAN_VAL
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.rstrip() for line in f]
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except IOError as e:
            print(f"Error: {e}")        

        for text in texts:
            clean_text , target = self.preprocessor(text)
            self.texts.append(clean_text)
            self.targets.append(target)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = self.texts[idx]
        if self.has_target:
            target = self.targets[idx]

        tokens = self.tokenizer(text)
        label = self.labels_dict[target]

        _input = torch.tensor(tokens)
        label = torch.tensor(label)

        return _input,label
        


def get_dataloader():
    pass


if __name__ == "__main__":
    dataset = ArmanDataset('val')
    print(dataset)