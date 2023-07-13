import torch
import numpy as np
import config
import hazm
import utils
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from preprocessing import Preprocessing
from transformers import AutoTokenizer

def get_data(train_path,test_path,cleaner):
    text_cleaner = cleaner
    train_df = pd.read_csv(train_path,encoding='utf-8')
    test_df = pd.read_csv(test_path,encoding='utf-8')
    train_df['text'] = train_df['text'].apply(text_cleaner)
    test_df['text'] = test_df['text'].apply(text_cleaner)
    X, y = train_df['text'], train_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test,y_test = test_df['text'] , test_df['label']
    X_train, y_train = X_train.values.tolist(), y_train.values.tolist()
    X_val, y_val = X_val.values.tolist(), y_val.values.tolist()
    X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
    return X_train,y_train,X_val,y_val,X_test,y_test

def test_data_handler(cleaner):
    df = pd.read_csv(config.TEST_FILE,encoding='utf-8')
    df = df[['tweet', 'primary_emotion']]
    df.columns = ['text', 'label']
    df.to_csv(config.MODIFIED_TEST, index=False)
    return df


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-base-parsbert-uncased')
        self.max_length = config.MAX_SEQ_LEN
        self.labels_dict = {'SAD': 0, 'HAPPY': 1,'SURPRISE': 2, 'FEAR': 3, 'HATE': 4, 'ANGRY': 5,'OTHER': 6,}

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
            'labels': torch.tensor(self.labels_dict[label])
        }
        return inputs



def create_dataloader(texts, labels,shuffle=True):
    dataset = TextDataset(texts, labels, max_length=config.MAX_SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
    return dataloader


# def handle_contest_dataset(path=None):
#     pass


# if __name__ == "__main__":
#     dataset = ArmanDataset('val')
#     print(dataset)