# from __future__ import unicode_literals

import torch
import pickle
import re

from torch.utils.data import Dataset,DataLoader
from vocabulary import Vocabulary
from configs import *
from hazm import *


class ArmanDataset(Dataset):
    def __init__(self, mode='train', vocab_path='vocab.pkl', vocab_threshold=5, vocab_from_file=False):
        super().__init__()
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_path = ARMAN_TEXT_TRAIN_PATH
        elif mode == 'test':
            self.data_path = ARMAN_TEXT_TEST_PATH

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()

        if vocab_from_file:
            self.vocab = self.load_vocab(vocab_path)
        else:
            self.vocab = Vocabulary(texts=self.texts, vocab_threshold=vocab_threshold)
            self.save_vocab(vocab_path)

        self.x_to_y = {}
        self.labels = {'sad':0,'hate':1,'fear':2,'angry':3,'happy':4,'surprise':5,'other':6}
        # self.label_idx = 0

        for text in self.texts:
            # tokens = word_tokenize(text)
            self.data_separator(text)

    def data_separator(self,text):
        tokens = word_tokenize(text)
        last_item = tokens[-1]
        if re.search('[a-zA-Z]', last_item):
            tokens.pop()

        # if last_item.lower() not in self.labels:
        #     self.labels[last_item.lower()] = self.label_idx
        #     self.label_idx += 1

        self.x_to_y[tuple(tokens)] = last_item.lower()

        return tokens

    def save_vocab(self, filepath):
        """
        Save the vocabulary to a file.

        Args:
        - filepath (str): Filepath to save the vocabulary.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab.word2index, f)

    def load_vocab(self, filepath):
        """
        Load a saved vocabulary from a file.

        Args:
        - filepath (str): Filepath to load the vocabulary from.
        """
        with open(filepath, 'rb') as f:
            my_dict = pickle.load(f)

        vocab = Vocabulary(texts=self.texts)
        vocab.word2index = my_dict
        vocab.index2word = {v: k for k, v in my_dict.items()}
        vocab.word2count = {k: 1 for k in my_dict.keys()}

        return vocab

    def __len__(self):
        return len(self.x_to_y)

    def __getitem__(self, index):
        tokens = list(self.x_to_y.keys())[index]
        label = self.x_to_y[tokens]

        # tokens = word_tokenize(tokens)
        indexed_tokens = [self.vocab.get_word_index(token) for token in tokens]
        # indexed_tokens = [idx for idx in indexed_tokens if idx not in [self.vocab.word2index['<UNK>'], self.vocab.word2index['<PAD>']]]

        # Padding
        padding_length = self.vocab.max_text_length - len(indexed_tokens)
        indexed_tokens += [self.vocab.get_word_index('<PAD>')] * padding_length

        # Convert to tensor

        indexed_tokens = torch.tensor(indexed_tokens)
        label = torch.tensor(self.labels[label])

        return indexed_tokens.unsqueeze(0), label


def create_dataloader(mode='train', vocab_path='vocab.pkl', vocab_threshold=5, vocab_from_file=False, batch_size=32, shuffle=True):
    dataset = ArmanDataset(mode=mode, vocab_path=vocab_path, vocab_threshold=vocab_threshold, vocab_from_file=vocab_from_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    train_dataloader = create_dataloader(mode='train', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=False, batch_size=32, shuffle=True)
    test_dataloader = create_dataloader(mode='test', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=True, batch_size=32, shuffle=False)
    print(train_dataloader.dataset.vocab.word2index)
    # print(len(test_dataloader.dataset.vocab.word2index))
    