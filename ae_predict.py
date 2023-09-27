import torch
from transformers import AutoTokenizer,AutoModel
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

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
    
model = XLMRobertaGRUClassifier(6)
model.load_state_dict(torch.load('xlmrobertalarge_gru_model.pth'))
model = model.cuda()

test_data = pd.read_csv('cleaned_final_test_data_emotion.csv')
model = model.cuda()
results = []
targets = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
for j in range(len(test_data)):
    data_point = test_data.iloc[j]
    text = data_point["tweet"]
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.cuda() for key, value in inputs.items()}  # Move tensors to CUDA

    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs)

    # Find the emotion with the highest predicted value
    primary_emotion_index = torch.argmax(probabilities)
    primary_emotion = targets[primary_emotion_index]

    # Normalize and apply threshold
    threshold = 0.15
    predictions = (probabilities > threshold).cpu().numpy().tolist()[0]

    # Create a dictionary with the required information
    result = {
        "local_id": data_point["local_id"],
        "tweet": text,
        "primary_emotion": primary_emotion,
    }

    # Add emotion predictions to the dictionary
    for i, emotion in enumerate(targets):
        result[emotion] = predictions[i]

    results.append(result)

# Create a DataFrame
result_df = pd.DataFrame(results)
result_df.to_csv('resulr.csv')

# Mapping from primary emotions
primary_emotion_mapping = {"Anger":'anger', "Sadness":'sadness', "Wonder":'surprise', "Happiness":'happiness', "Fear":'fear', "Hatred":'disgust', "Other":'other'}

# Tokenize and predict emotions for each tweet
results = []

for j in range(len(test_data)):
    data_point = test_data.iloc[j]
    text = data_point["tweet"]
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.cuda() for key, value in inputs.items()}  # Move tensors to CUDA

    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs)

    # Find the emotion with the highest predicted value
    primary_emotion_index = torch.argmax(probabilities)
    primary_emotion = primary_emotion_mapping[targets[primary_emotion_index]]

    # Normalize and apply threshold
    threshold = 0.15
    predictions = (probabilities > threshold).cpu().numpy().tolist()[0]

    # Create a dictionary with the required information
    result = {
        "local_id": data_point["local_id"],
        "tweet": text,
        "primary_emotion": primary_emotion,
    }

    # Add emotion predictions to the dictionary
    for i, emotion in enumerate(targets):
        # Convert True/False to 1/0
        result[emotion] = int(predictions[i])

    # If all predicted scores are 0, set 'other' as primary emotion
    if all(score == 0 for score in predictions):
        result["primary_emotion"] = "Other"

    results.append(result)

# Create a DataFrame
result_df = pd.DataFrame(results)
result_df['primary_emotion'] = result_df['primary_emotion'].replace({'Other': 'other'})
result_df = result_df.rename(columns={"Anger":'anger', "Sadness":'sadness', "Wonder":'surprise', "Happiness":'happiness', "Fear":'fear', "Hatred":'disgust'})
cols = ['local_id', 'tweet', 'primary_emotion', 'anger', 'disgust', 'fear', 'sadness', 'happiness', 'surprise']
result_df1 = result_df[cols]
result_df1.to_csv('large_gru_AE.csv')
