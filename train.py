import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup


batch_size = 16
learning_rate = 1e-5
num_epochs = 5

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
test = pd.read_csv('test_cleaned_emopars.csv', index_col=0)
train = pd.read_csv('train_cleaned_emopars.csv', index_col=0)
targets = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]
def preprocess(df):
    # Normalize emotion labels
    emotion_columns = ['Anger', 'Fear', 'Happiness', 'Hatred', 'Sadness', 'Wonder']

    for col in emotion_columns:
        df[col] = df[col] / df[col].max()  # Normalize to the range [0, 1]

    # Apply threshold for binary labels
    threshold = 0.35
    for col in emotion_columns:
        df[col] = df[col].apply(lambda x: 1 if x >= threshold else 0)

    return df
train = pd.concat([train,test])
train = preprocess(train)
test = preprocess(test)
X_train, y_train = train['text'].values.tolist(), train[targets].values.tolist()
X_test, y_test = test['text'].values.tolist(), test[targets].values.tolist()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Use the provided numeric label directly
        }

        return inputs

train_dataset = TextDataset(tokenizer,X_train,y_train,max_length=128)
test_dataset = TextDataset(tokenizer,X_test,y_test,max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
import torch.nn as nn

# Define the loss function
criterion = nn.BCEWithLogitsLoss()
# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel

class XLMRobertaGRUClassifier(nn.Module):
    def __init__(self, num_classes):
        super(XLMRobertaGRUClassifier, self).__init__()
        self.num_classes = num_classes
        self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-large")

        # Add a GRU layer
        self.gru = nn.GRU(self.xlmroberta.config.hidden_size, hidden_size=self.xlmroberta.config.hidden_size, num_layers=1, batch_first=True)

        # # Correct the hidden size for the linear layer
        self.linear = nn.Linear(self.xlmroberta.config.hidden_size, num_classes)  # Multiply by 2 for bidirectional GRU
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmroberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.last_hidden_state

        # Pass the logits through the GRU layer
        gru_output, _ = self.gru(logits)

        logits = self.linear(self.dropout(gru_output[:, -1, :]))
        return logits
    
epochs = 5
num_classes = 6
model = XLMRobertaGRUClassifier(num_classes)
model.to(device)  # Move model to GPU if available
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
from tqdm import tqdm
for epoch in range(epochs):
    model.train()
    total_loss = 0
    idx = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if idx % 20 == 0:
            print(f"batch loss: {loss.item():.4f}")
        idx +=1

    average_loss = total_loss / len(train_dataloader)
    print('-' * 80)
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")

# Save your trained model
    torch.save(model.state_dict(), 'xlmrobertalarge_gru_model.pth')
