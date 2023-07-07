
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

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

# # Example usage
# vocab = {'apple': 1, 'banana': 2, 'orange': 3}
# save_vocab(vocab, 'vocab.pkl')

# loaded_vocab = load_vocab('vocab.pkl')
# print(loaded_vocab)
