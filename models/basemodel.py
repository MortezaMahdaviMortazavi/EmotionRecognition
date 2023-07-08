import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("forward() must be implemented in the derived class.")

    def shared_layers(self):
        raise NotImplementedError("shared_layers() must be implemented in the derived class.")

    def classifier(self, features):
        raise NotImplementedError("classifier() must be implemented in the derived class.")
