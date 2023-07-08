import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import AutoModel
from basemodel import BaseModel


finetunes_folder = "../fintuned_models/"
model_file = "ParsBERT.pt"
bert_path = os.path.join(finetunes_folder, model_file)



class BiGruBERT(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        pass
    
    def forward(self, input_ids, attention_mask):
        return super().forward(input_ids, attention_mask)
    
    
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BidirectionalGRUAttentionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = torch.load(bert_path)
        self.bert.classifier = nn.Identity()
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_size=128, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * 128, 1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2 * 128, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        print("outputs: ",outputs)
        last_hidden_state = outputs[0]  # Access the correct attribute for hidden states
        print("last hidden state size: ",last_hidden_state)
        gru_output, _ = self.gru(last_hidden_state)
        print("gru output shape: ",gru_output.shape)
        attention_weights = self.attention(gru_output).squeeze(-1)
        print("attn res: ",attention_weights.shape)
        attention_weights = torch.softmax(attention_weights, dim=1).unsqueeze(-1)
        print("attn after softmax",attention_weights.shape)
        attention_output = torch.sum(gru_output * attention_weights, dim=1)
        print("attn output",attention_output.shape)
        attention_output = self.dropout(attention_output)
        logits = self.fc(attention_output)
        print("logits shape",logits.shape)
        return logits

# Create an instance of the model
model = BidirectionalGRUAttentionClassifier(num_classes=10)
# Create an instance of the model
# model = BidirectionalGRUAttentionClassifier(num_classes=10)

import torch
from transformers import AutoTokenizer

# Text input
text = "سلام، حال شما چطور است؟"

# Tokenization using ParsBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Padding and attention mask
max_seq_length = 128  # Maximum sequence length expected by the model
padding_length = max_seq_length - len(input_ids)

if padding_length > 0:
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length

attention_mask = [1] * len(input_ids)

# Convert to tensors
input_ids = torch.tensor([input_ids])
attention_mask = torch.tensor([attention_mask])

# Create an instance of the model
model = BidirectionalGRUAttentionClassifier(num_classes=10)

# Example usage
logits = model(input_ids, attention_mask)
print(logits)
