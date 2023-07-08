import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import AutoModel
from basemodel import BaseModel



finetunes_folder = "../fintuned_models/"
model_file = "ParsBERT.pt"
bert_path = os.path.join(finetunes_folder, model_file)

class ParsBERT(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # self.bert = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.bert = torch.load(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert.classifier = nn.Linear(768,7)
        for param in self.bert.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = self.fc(pooled_output)
        return logits

    def shared_layers(self):
        # No shared layers in this model
        pass

    def classifier(self, features):
        # No additional classifier layers in this model
        pass

