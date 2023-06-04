from __future__ import unicode_literals


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from transformers import AutoTokenizer, AutoModelForTokenClassification
# from dataloader import create_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from configs import *
from tqdm import tqdm
from embedding import Embedding,CustomEmbedding,TransformersTokenizer


class ParsBERT(nn.Module):
    def __init__(self):
        super().__init__()
        """In this section we Implement ParsBERT for finetuning on task"""
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-ner-uncased")

        self.model = AutoModelForTokenClassification.from_pretrained("HooshvareLab/bert-base-parsbert-ner-uncased")
        """The Implementation of ParsBERT is finished"""

    # def model_changer(self):
    

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.embedding = Embedding(vocab_size=vocab_size,output_size=hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.embedding(x.long()).squeeze(1))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        out = self.relu(self.fc(out[:, -1, :]))  # Use the last time step's output for classification
        out = F.dropout(out,p=0.2)
        # out = F.softmax(out,dim=1)
        return out

