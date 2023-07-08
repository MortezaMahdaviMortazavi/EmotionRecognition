import torch
import torch.nn as nn
import torch.nn.functional as F

import config

from train import train
from models.parsbert import ParsBERT
from models.gru_bert import BiGruBERT

class ModelFactory:
    def create_model(self, model_type):
        if model_type == "ParsBERT":
            return ParsBERT(num_classes=config.NUM_CLASSES)
        elif model_type == "BiGruBERT":
            return BiGruBERT(num_classes=config.NUM_CLASSES)
        else:
            raise ValueError("Invalid model type.")
        

if __name__ == "__main__":
    # Client Code
    model_factory = ModelFactory()

    resnet_model = model_factory.create_model("resnet")
    print(resnet_model)  # Output: ResNet()

    mobilenet_model = model_factory.create_model("mobilenet")
    print(mobilenet_model)  # Output: MobileNet()
