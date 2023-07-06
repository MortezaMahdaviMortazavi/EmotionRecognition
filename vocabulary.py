import os
import pandas as pd
import re
import pickle
import hazm
import config
import tqdm

from preprocessing import Preprocessing

class Vocabulary:
    def __init__(self,texts,vocab_threshold=3,name='arman'):
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
        self.max_text_length = config.MAX_LEN


        self.preprocessor = Preprocessing(dataset=name)
        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1
        self.index2word[0] = '<PAD>'
        self.build_vocab(texts=texts)
        self.labels = self.preprocessor.get_labels()


    def __call__(self,word):
        return self.word2index[word]

    def __repr__(self) -> str:
        return f"Labels : {self.labels} | length of vocab : {len(self.word2index)}"

    def build_vocab(self, texts):
        """
        Build the vocabulary based on the provided texts.

        Args:
        - texts (list): List of text samples to build the vocabulary from.
        """
        for text in tqdm.tqdm(texts):
            cleaned_text = self.preprocessor(text)
            tokens = hazm.word_tokenize(cleaned_text)
            self.add_words(tokens)


    def add_word(self, word):
        """
        Add a word to the vocabulary.

        Args:
        - word (str): Word to be added to the vocabulary.
        """
        # word = self.lemmatizer.lemmatize(word)
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

    def get_labels(self):
        return self.labels




if __name__ == "__main__":

    # PERSIAN_EMOTION_DATASET = pd.read_csv(config.ARMAN_TRAIN,encoding='utf-8')

    with open(config.ARMAN_VAL,'r',encoding='utf-8') as f:
        texts = f.readlines()

    vocab = Vocabulary(texts=texts,vocab_threshold=3,name='arman')
    print(vocab)
    