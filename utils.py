import csv
import pandas as pd
import pickle
import torch

from sklearn.model_selection import train_test_split


def tsv2csv(tsv_file,csv_file):
    # Open the TSV file for reading
    with open(tsv_file, "r", encoding="utf-8") as tsv:
        # Create a CSV file for writing
        with open(csv_file, "w", encoding="utf-8", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["text", "label"])  # Write the header row

            # Read each line in the TSV file and convert it to CSV format
            for line in tsv:
                line = line.strip().split("\t")
                text = line[0]
                label = line[1]

                writer.writerow([text, label])  # Write the row in CSV format


def read_data(address):
    cleaning = None
    targets = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]
    target = targets[0]
    df = pd.read_csv(address, usecols=["text", target])
    
    df['text'] = df['text'].apply(cleaning)
    df = df.sample(frac=1).reset_index(drop=True)

    train, test = train_test_split(df, test_size=0.1, random_state=1, stratify=df[target])
    train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train[target])

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)

    x_train, y_train = train['text'].values.tolist(), train[target].values.tolist()
    x_valid, y_valid = valid['text'].values.tolist(), valid[target].values.tolist()
    x_test, y_test = test['text'].values.tolist(), test[target].values.tolist()

    return train, valid, test, x_train, y_train, x_valid, y_valid, x_test, y_test



def write_text_to_file(text,file_path):
    with open(file_path, 'a',encoding='utf-8') as file:
        file.write(text)
    print(f"Text written to file: {file_path}")



def save_vocab(vocab, path):
    with open(path, 'wb') as file:
        pickle.dump(vocab, file)
    print(f"Vocabulary saved to: {path}")

def load_vocab(path):
    with open(path, 'rb') as file:
        vocab = pickle.load(file)
    return vocab

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    train_f1 = checkpoint['train_f1_score']
    val_loss = checkpoint['valid_loss']
    val_acc = checkpoint['valid_acc']
    val_f1 = checkpoint['valid_f1_score']
    
    return epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1

