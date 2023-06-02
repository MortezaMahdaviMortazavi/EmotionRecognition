import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import LSTMModel,Embedding
from dataloader import create_dataloader

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def fit(self, train_loader, valid_loader, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
            train_loss, train_acc = self.train_one_epoch(train_loader)
            valid_loss, valid_acc = self.evaluate(valid_loader)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accuracies.append(train_acc)
            self.valid_accuracies.append(valid_acc)
            print(f"Train Loss: {train_loss:.4f} Accuracy: % {train_acc * 100:.4f}", end=" | ")
            print(f"Valid Loss: {valid_loss:.4f} Accuracy: % {valid_acc * 100:.4f}")

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.float().to(self.device)
            labels = labels.long().to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy


def main():
    # Prepare the data
    train_loader = create_dataloader(mode='train', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=False, batch_size=64, shuffle=True)
    test_loader = create_dataloader(mode='test', vocab_path='vocab.pkl', vocab_threshold=10, vocab_from_file=True, batch_size=16, shuffle=False)
    # Define hyperparameters and model
    input_size = len(train_loader.dataset.vocab.word2index)  # Input size based on your data
    hidden_size = 105
    num_layers = 2
    num_classes = 7  # Number of classes based on your data

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop
    num_epochs = 1000

        # Instantiate the Trainer
    trainer = Trainer(model, criterion, optimizer, device)

    # Call the fit function
    trainer.fit(train_loader, test_loader, num_epochs)



if __name__ == "__main__":
    main()