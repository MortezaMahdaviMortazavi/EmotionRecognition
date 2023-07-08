import pytorch_lightning as pl
import torch

from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
import utils

class LightningTrainer(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, device, log_file):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_file = log_file

    def forward(self, x, attention_mask):
        return self.model(x, attention_mask=attention_mask)

    def common_step(self, batch, step_name):
        inputs = batch["input"].to(self.device)
        attention_masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.model(inputs, attention_mask=attention_masks)
        loss = self.criterion(outputs, labels)

        _, predicted = outputs.max(1)
        accuracy = torch.sum(predicted == labels).item() / labels.size(0)

        self.log(f'{step_name}_loss', loss, on_step=step_name == "train", on_epoch=True, prog_bar=True)
        self.log(f'{step_name}_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Log the details in a text file
        with open(self.log_file, 'a') as f:
            f.write(f'{step_name.capitalize()} Step: {self.global_step}\n')
            f.write(f'Loss: {loss.item():.4f}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write('\n')

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, step_name="test")

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        # Implement your own train data loader here
        train_dataset = ...
        train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)
        return train_loader

    def val_dataloader(self):
        # Implement your own validation data loader here
        val_dataset = ...
        val_loader = DataLoader(val_dataset, batch_size=..., shuffle=False)
        return val_loader

    def test_dataloader(self):
        # Implement your own test data loader here
        test_dataset = ...
        test_loader = DataLoader(test_dataset, batch_size=..., shuffle=False)
        return test_loader

        
def train():
    # Instantiate the LightningTrainer
    model = ...
    criterion = ...
    optimizer = ...
    device = ...
    logger = TensorBoardLogger('logs/', name='my_model')
    trainer = pl.Trainer(
        callbacks=[ModelCheckpoint(dirpath='checkpoints', filename='model-{epoch:02d}-{val_loss:.2f}', save_top_k=3)],
        max_epochs=config.MAX_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # Train the model
    lightning_trainer = LightningTrainer(model, criterion, optimizer, device)
    trainer.fit(lightning_trainer)




if __name__ == "__main__":
    train()