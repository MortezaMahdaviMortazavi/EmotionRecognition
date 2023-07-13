import pytorch_lightning as pl
import torch

from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score, accuracy_score

from dataset import ArmanDataset,create_dataloader
from tqdm import tqdm

import config
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

# class Trainer:
#     def __init__(self, model, criterion, optimizer, device):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device
#         self.log_file = config.LOGFILE
#         self.train_losses = []
#         self.valid_losses = []
#         self.train_accuracies = []
#         self.valid_accuracies = []
#         self.train_f1_scores = []
#         self.valid_f1_scores = []

#     def fit(self, train_loader, valid_loader, num_epochs):
#         best_val_acc = 0
#         for epoch in tqdm(range(num_epochs)):
#             print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
#             train_loss, train_acc, train_f1_score = self.train_one_epoch(train_loader)
#             valid_loss, valid_acc, valid_f1_score = self.evaluate(valid_loader)
#             self.train_losses.append(train_loss)
#             self.valid_losses.append(valid_loss)
#             self.train_accuracies.append(train_acc)
#             self.valid_accuracies.append(valid_acc)
#             self.train_f1_scores.append(train_f1_score)
#             self.valid_f1_scores.append(valid_f1_score)
#             print(f"Train Loss: {train_loss:.4f} Accuracy: {train_acc * 100:.4f}", end=" | ")
#             print(f"Train F1 Score: {train_f1_score:.4f}", end=" | ")
#             print(f"Valid Loss: {valid_loss:.4f} Accuracy: {valid_acc * 100:.4f}", end=" | ")
#             print(f"Valid F1 Score: {valid_f1_score:.4f}")
#             with open(self.log_file, 'a') as f:
#                 f.write(f"Epoch {epoch+1}/{num_epochs} | ")
#                 f.write(f"Train Loss: {train_loss:.4f} | ")
#                 f.write(f"Train Accuracy: {train_acc:.4f} | ")
#                 f.write(f"Train F1 Score: {train_f1_score:.4f} | ")
#                 f.write(f"Valid Loss: {valid_loss:.4f} | ")
#                 f.write(f"Valid Accuracy: {valid_acc:.4f} | ")
#                 f.write(f"Valid F1 Score: {valid_f1_score:.4f} | ")
#                 f.write("\n")

#             # Save the model if validation accuracy improves
#             if valid_acc > best_val_acc:
#                 best_val_acc = valid_acc
#                 checkpoint = {
#                     'epoch': epoch + 1,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                     'train_loss': train_loss,
#                     'train_acc': train_acc,
#                     'train_f1_score': train_f1_score,
#                     'valid_loss': valid_loss,
#                     'valid_acc': valid_acc,
#                     'valid_f1_score': valid_f1_score
#                 }
#                 torch.save(checkpoint, 'checkpoints/model_checkpoint.pth')

#     def train_one_epoch(self, train_loader):
#         self.model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
#         pred_labels = []
#         true_labels = []

#         for sample in tqdm(train_loader):
#             inputs = sample['input']
#             labels = sample['label']
#             masks = sample['mask'].to(self.device) 

#             inputs = inputs.to(self.device)
#             labels = labels.to(self.device)

#             self.optimizer.zero_grad()

#             outputs = self.model(inputs, masks)
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()

#             total_loss += loss.item()

#             _, predicted = outputs.max(1)
#             pred_labels.extend(predicted.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())

#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)

#         avg_loss = total_loss / len(train_loader)
#         accuracy = accuracy_score(true_labels, pred_labels)
#         f1_score_value = f1_score(true_labels, pred_labels, average='macro')

#         return avg_loss, accuracy, f1_score_value

#     def evaluate(self, data_loader):
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
#         pred_labels = []
#         true_labels = []

#         with torch.no_grad():
#             for sample in tqdm(data_loader):
#                 inputs = sample['input']
#                 labels = sample['label']
#                 masks = sample['mask'].to(self.device) 

#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)

#                 outputs = self.model(inputs, masks)
#                 loss = self.criterion(outputs, labels)

#                 total_loss += loss.item()

#                 _, predicted = outputs.max(1)
#                 pred_labels.extend(predicted.cpu().numpy())
#                 true_labels.extend(labels.cpu().numpy())

#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.size(0)

#         avg_loss = total_loss / len(data_loader)
#         accuracy = accuracy_score(true_labels, pred_labels)
#         f1_score_value = f1_score(true_labels, pred_labels, average='macro')

#         return avg_loss, accuracy, f1_score_value

from sklearn.metrics import f1_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, learning_rate=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = self._train_epoch()
            val_loss, val_acc, val_f1 = self._evaluate(self.val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Train F1 Score: {train_f1:.4f}")
            print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Validation F1 Score: {val_f1:.4f}")
            print()

            with open(config.LOGFILE, 'a') as f:
                f.write(f"Epoch {epoch+1}/{num_epochs} | ")
                f.write(f"Train Loss: {train_loss:.4f} | ")
                f.write(f"Train Accuracy: {train_acc:.4f} | ")
                f.write(f"Train F1 Score: {train_f1:.4f} | ")
                f.write(f"Valid Loss: {val_loss:.4f} | ")
                f.write(f"Valid Accuracy: {val_acc:.4f} | ")
                f.write(f"Valid F1 Score: {val_f1:.4f} | ")
                f.write("\n")
                
        checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1_score': train_f1,
                    'valid_loss': val_loss,
                    'valid_acc': val_acc,
                    'valid_f1_score': val_f1
                }
        torch.save(checkpoint, 'checkpoints/model_checkpoint.pth')

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_predictions = []
        total_labels = []

        for batch in tqdm(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)

            loss = self.criterion(outputs.logits, labels)
            total_loss += loss.item()

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            total_predictions.extend(predicted.cpu().tolist())
            total_labels.extend(labels.cpu().tolist())

            loss.backward()
            self.optimizer.step()

        average_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        f1 = f1_score(total_labels, total_predictions, average='macro')
        return average_loss, accuracy, f1

    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_predictions = []
        total_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, dim=1)

                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                total_predictions.extend(predicted.cpu().tolist())
                total_labels.extend(labels.cpu().tolist())

        average_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples
        f1 = f1_score(total_labels, total_predictions, average='macro')

        
        return average_loss, accuracy, f1



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
    num_epochs = 100

    trainer = Trainer(model=_model,criterion=criterion,optimizer=optimizer,device=device)
    # Call the fit function
    trainer.fit(train_loader,val_loader,num_epochs=num_epochs)



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
