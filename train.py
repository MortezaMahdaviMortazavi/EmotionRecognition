import pytorch_lightning as pl
import torch

from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from dataset import ArmanDataset,create_dataloader
from tqdm import tqdm

import config
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_file = config.LOGFILE
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def fit(self, train_loader, valid_loader, num_epochs):
        best_val_acc = 0
        for epoch in tqdm(range(num_epochs)):
            print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
            train_loss, train_acc , train_f1_score = self.train_one_epoch(train_loader)
            valid_loss, valid_acc , val_f1_score = self.evaluate(valid_loader)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accuracies.append(train_acc)
            self.valid_accuracies.append(valid_acc)
            print(f"Train Loss: {train_loss:.4f} Accuracy: % {train_acc * 100:.4f}", end="  ")
            print(f"Train F1 Score: {train_f1_score:.4f}",end=" | ")
            print(f"Valid Loss: {valid_loss:.4f} Accuracy: % {valid_acc * 100:.4f}")
            print(f"Val F1 Score: {val_f1_score:.4f}",end=" | ")
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch} | ")
                f.write(f'Train Loss: {train_loss:.4f} | ')
                f.write(f'Train Accuracy: {train_acc:.4f} | ')
                f.write(f'Train f1_score: {train_f1_score:.4f} | ')
                f.write(f'Val Loss: {valid_loss:.4f} | ')
                f.write(f'Val Accuracy: {valid_acc:.4f} | ')
                f.write(f'Val f1_score: {val_f1_score:.4f} | ')
                f.write('\n')

            # Save the model if validation accuracy improves
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1_score': train_f1_score,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'val_f1_score': val_f1_score
                }
                torch.save(checkpoint, 'checkpoints/model_checkpoint.pth')

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        _f1_point = 0

        for sample in train_loader:
            inputs = sample['input']
            labels = sample['label']
            masks = sample['mask'].to(self.device) 

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs,masks)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(),average='macro')
            _f1_point += f1
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        _f1_point/=len(train_loader)
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy , _f1_point

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        _f1_point = 0

        with torch.no_grad():
            for sample in data_loader:
                inputs = sample['input']
                labels = sample['label']
                masks = sample['mask'].to(self.device) 
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs,masks)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
            
                f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(),average='macro')
                _f1_point += f1

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        _f1_point/=len(data_loader)
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy,_f1_point


def main(model):
    # Prepare the data
    train_loader = create_dataloader(dataset_type='train',is_preprocess=True,load_vocab=True,shuffle=False)
    val_loader = create_dataloader(dataset_type='val',is_preprocess=True,shuffle=True)
    # Define hyperparameters and model

    _model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(_model.parameters(), lr=0.003)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop
    num_epochs = 1000

    trainer = Trainer(_model, criterion, optimizer, device)

    # Call the fit function
    trainer.fit(train_loader, val_loader, num_epochs)



# if __name__ == "__main__":
#     main()
        
# def train(model):
#     # Instantiate the LightningTrainer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
#     device = config.DEVICE
#     logger = TensorBoardLogger('logs/', name='my_model')
#     trainer = pl.Trainer(
#         callbacks=[ModelCheckpoint(dirpath='checkpoints', filename='model-{epoch:02d}-{val_loss:.2f}', save_top_k=3,monitor='val_loss')],
#         max_epochs=config.MAX_EPOCHS,
#     )

#     # Train the model
#     lightning_trainer = LightningTrainer(model, criterion, optimizer)
#     trainer.fit(lightning_trainer)
