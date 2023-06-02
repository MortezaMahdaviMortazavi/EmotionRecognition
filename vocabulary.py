import os
import pandas as pd
import re
import pickle

from hazm import *
from configs import *

class Vocabulary:
    def __init__(self,texts,vocab_threshold=3,name='persian'):
        """
        Initialize the Vocabulary class.

        Args:
        - name (str): Name of the vocabulary, default is 'persian'.
        """
        self.name = name
        self.vocab_threshold = vocab_threshold
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.max_text_length = 105
        
        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1
        self.index2word[0] = '<PAD>'
        self.build_vocab(texts=texts)
        


    def remove_last_if_english(self,lst):
        last_item = lst[-1]
        if re.search('[a-zA-Z]', last_item):
            lst.pop()

        return lst

    def build_vocab(self, texts):
        """
        Build the vocabulary based on the provided texts.

        Args:
        - texts (list): List of text samples to build the vocabulary from.
        """
        for text in texts:
            words = word_tokenize(text)

            words = self.remove_last_if_english(words)
            if len(words) > self.max_text_length:
                self.max_text_length = len(words)

            self.add_words(words)

    def add_word(self, word):
        """
        Add a word to the vocabulary.

        Args:
        - word (str): Word to be added to the vocabulary.
        """
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

        if word not in self.word2index and self.word2count[word] >= self.vocab_threshold:
            self.word2index[word] = len(self.word2index)
            self.index2word[len(self.word2index)-1] = word

    def add_words(self, words):
        """
        Add a list of words to the vocabulary.

        Args:
        - words (list): List of words to be added to the vocabulary.
        """
        for word in words:
            self.add_word(word)

    def get_word_index(self, word):
        """
        Get the index of a word in the vocabulary.

        Args:
        - word (str): Word to retrieve the index for.

        Returns:
        - index (int): Index of the word in the vocabulary, -1 if the word is not present.
        """
        return self.word2index.get(word,self.word2index['<UNK>'])

    def get_index_word(self, index):
        """
        Get the word corresponding to a given index in the vocabulary.

        Args:
        - index (int): Index to retrieve the word for.

        Returns:
        - word (str): Word corresponding to the given index, None if the index is out of range.
        """
        if 0 <= index < len(self.index2word):
            return self.index2word[index]
        return None

    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
        - size (int): Size of the vocabulary.
        """
        return len(self.word2index)



    def __call__(self,word):
        self.word2index[word]



if __name__ == "__main__":

    PERSIAN_EMOTION_DATASET = pd.read_csv(PERSIAN_EMOTION_PATH,encoding='utf-8')
    INTRO_DATASET = pd.read_csv(INTRO_DATA_PATH,encoding='utf-8')

    with open(ARMAN_TEXT_TRAIN_PATH,'r',encoding='utf-8') as f:
        texts = f.readlines()

    vocab = Vocabulary(texts=texts,vocab_threshold=8)
    print(vocab.word2index)