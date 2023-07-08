import torch
import numpy as np
import config
import hazm

from tqdm import tqdm
from preprocessing import Preprocessing
from vocabulary import Vocabulary

class ArmanDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_type,
            tokenizer_type='hazm',
            load_vocab=True   
        ):
        assert dataset_type in ['train', 'val']
        assert tokenizer_type in ['hazm', 'parsbert']
        self.dataset_type = dataset_type
        self.tokenizer_type = tokenizer_type
        self.texts = [] # the text in each sample of dataset
        self.targets = [] # the labels or targets of each sample
        self.tokenizer = config.TOKENIZER # tokenizer that tokenize the text
        self.preprocessor = Preprocessing(dataset='arman') # return text and its target while we call it with a text input
        self.has_target = isinstance(self.targets, list) or isinstance(self.targets, np.ndarray)


        self.labels_dict = {'SAD': 0, 'HAPPY': 1, 'OTHER': 2, 'SURPRISE': 3, 'FEAR': 4, 'HATE': 5, 'ANGRY': 6}
        self.vocab = Vocabulary(self.texts,vocab_threshold=2,name='arman',load=load_vocab)
        self.extract_data()
    
    def extract_data(self):
        file_path = config.ARMAN_TRAIN if self.dataset_type == 'train' else config.ARMAN_VAL
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.rstrip() for line in tqdm(f)]
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except IOError as e:
            print(f"Error: {e}")        

        for text in tqdm(texts[:50]):
            clean_text , target = self.preprocessor(text)
            self.texts.append(clean_text)
            self.targets.append(target)

    def __len__(self):
        return len(self.texts)
    
    def __repr__(self):
        return f"Sample: {self[1]}"
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.has_target:
            target = self.labels_dict[self.targets[idx]]

        tokenized_text = self.tokenizer.tokenize(text)

        if len(tokenized_text) < config.MAX_SEQ_LEN:
            tokenized_text += ['[PAD]'] * (config.MAX_SEQ_LEN - len(tokenized_text))
        else:
            tokenized_text = tokenized_text[:config.MAX_SEQ_LEN]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        attention_mask = [1] * len(input_ids)

        # Convert to tensors
        padded_input_ids = torch.tensor(input_ids).unsqueeze(0)
        padded_attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        label = torch.tensor(target)

        _instance = {
            "input":padded_input_ids,
            "label":label,
            "mask":padded_attention_mask
        }

        return _instance


def create_dataloader(
    dataset_type,
    tokenizer_type='hazm',
    load_vocab=True,
    batch_size=32,
    shuffle=True,
    num_workers=0
):
    dataset = ArmanDataset(
        dataset_type=dataset_type,
        tokenizer_type=tokenizer_type,
        load_vocab=load_vocab
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    dataloader = create_dataloader('val',)
    print(len(dataloader.dataset))
