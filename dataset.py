import torch
import numpy as np
import config

from preprocessing import Preprocessing
from vocabulary import Vocabulary

class ArmanDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_type):
        assert dataset_type in ['train','val']
        self.type = dataset_type # train or val
        self.texts = [] # the text in each sample of dataset
        self.targets = [] # the labels or targets of each sample
        self.tokenizer = config.TOKENIZER # tokenizer that tokenize the text
        self.preprocessor = Preprocessing(dataset='arman') # return text and its target while we call it with a text input
        self.has_target = isinstance(self.targets, list) or isinstance(self.targets, np.ndarray)


        self.labels_dict = self.preprocessor.labels
        self.vocab = Vocabulary(self.texts,vocab_threshold=2,name='arman')
        self.extract_data()
    
    def extract_data(self):
        
        if self.type == 'train':
            with open(config.ARMAN_TRAIN,'r',encoding='utf-8') as f:
                texts = f.readlines()
            
        elif self.type == 'val':
            with open(config.ARMAN_VAL,'r',encoding='utf-8') as f:
                texts = f.readlines()

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
        




