from __future__ import unicode_literals


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataloader import create_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader


class ParsBERT(nn.Module):
    """In this section we Implement ParsBERT for finetuning on task"""
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-ner-uncased")

    model = AutoModelForTokenClassification.from_pretrained("HooshvareLab/bert-base-parsbert-ner-uncased")
    """The Implementation of ParsBERT is finished"""



class Embedding(nn.Module):
    def __init__(self,vocab_size,output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,output_size)
        self.linear = nn.Linear(output_size,vocab_size)

    def forward(self,x):
        output = self.embedding(x)
        return F.dropout(self.linear(output),p=0.3)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = self.embedding(x.long()).squeeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last time step's output for classification
        out = F.dropout(out,p=0.2)
        # out = F.softmax(out,dim=1)
        return out

# # Prepare the data
# train_loader = create_dataloader(mode='train', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=False, batch_size=64, shuffle=True)
# test_loader = create_dataloader(mode='test', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=True, batch_size=16, shuffle=False)
# # Define hyperparameters and model
# input_size = len(train_loader.dataset.vocab.word2index)  # Input size based on your data
# print("input size shape is",input_size)
# hidden_size = 105
# num_layers = 2
# num_classes = 7  # Number of classes based on your data

# model = LSTMModel(input_size, hidden_size, num_layers, num_classes).cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.005)

# # Training loop
# num_epochs = 100

# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     train_loss = 0.0
#     correct = 0
#     total = 0

#     for inputs, labels in train_loader:
#         inputs = inputs.float().cuda()
#         labels = labels.long().cuda()
#         # print("inputs shape is",inputs.shape)
#         # print("labels shape is",labels.shape)
#         # print("embedding shape is",model.embedding(inputs.squeeze(1).long()).shape)
#         # break

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)

#         correct += (predicted == labels).sum().item()
    
#     # break

#     average_train_loss = train_loss / len(train_loader)
#     train_accuracy = 100 * correct / total

#     # Evaluation
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.float().cuda()
#             labels = labels.long().cuda()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()

#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     average_test_loss = test_loss / len(test_loader)
#     test_accuracy = correct / total

#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {average_train_loss:.4f} | Train Accuracy: % {train_accuracy:.4f} |Test Loss: {average_test_loss:.4f} | Test Accuracy: % {test_accuracy*100:.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "lstm_model.pth")
