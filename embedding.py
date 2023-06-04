import pickle
import torch
import torch.nn as nn

from configs import *
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModelForTokenClassification


class CustomEmbedding(nn.Module):
    def __init__(self, word_vectors_path=EMBEDDING_FILE):
        super(CustomEmbedding, self).__init__()
        # Load the word embeddings
        word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=False)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors.vectors), padding_idx=0,freeze=True)

    def forward(self, x):
        return self.embedding(x)
    

class Embedding(nn.Module):
    def __init__(self,vocab_size,output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,output_size)
        self.linear = nn.Linear(output_size,output_size)

    def forward(self, x):
        output = self.embedding(x)
        # output = F.dropout(output, p=0.2)  # Apply dropout to the output of the embedding
        # output = self.linear(output)
        return output


class TransformersTokenizer:
    def __init__(self, model_name='HooshvareLab/bert-base-parsbert-ner-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_pad(self, input_texts, max_length=None):
        # Tokenize the input texts
        tokenized_texts = self.tokenizer(input_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

        return tokenized_texts['input_ids']

# def set_transformer_tokenizer(self):
#     model_name='HooshvareLab/bert-base-parsbert-ner-uncased'
#     self.tokenizer = AutoTokenizer.from_pretrained(model_name)



def pretrained_embedding_handler(saved_embedding_file,embedding_file):
    # Load word embeddings if not already saved
    try:
        # Try loading saved embeddings
        with open(saved_embedding_file, 'rb') as file:
            word_vectors = pickle.load(file)
            print("Saved embeddings loaded successfully!")

    except FileNotFoundError:
        # Load word embeddings from the original file
        word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=False)

        # Save the loaded embeddings
        with open(saved_embedding_file, 'wb') as file:
            pickle.dump(word_vectors, file)
            print("Embeddings saved for future use!")







if __name__ == "__main__":
    saved_embedding_file = SAVED_EMBEDDING_FILE
    embedding_file = EMBEDDING_FILE

    # Initialize the TransformersTokenizer
    tokenizer = TransformersTokenizer()

    # Define the input text
    input_text = "خیلی کوچیک هستن و سایزشون بدرد نمیخوره میخوام پس بدم"

    # Tokenize and pad the input text
    input_ids = tokenizer.tokenize_and_pad([input_text],max_length=105)

    # Print the resulting input IDs tensor
    print(input_ids.shape,input_ids)