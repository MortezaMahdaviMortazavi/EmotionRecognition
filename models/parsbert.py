import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import AutoModel
# from basemodel import BaseModel



# Get the absolute path to the ParsBERT.pt file
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ParsBERT.pt")

class ParsBERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        try:
            self.bert = torch.load(file_path)
        except:
            self.bert = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.bert.classifier = nn.Identity()
        self.fc = nn.Linear(768,7)
        for param in self.bert.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.logits
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = self.fc(pooled_output)
        # print("\n\nlogits max 1: ",logits.max(1),"\n\n")
        return logits

