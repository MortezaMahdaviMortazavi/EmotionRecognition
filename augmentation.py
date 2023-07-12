import pandas as pd
import nlpaug.augmenter.word as naw
import config
import utils

from tqdm import tqdm

def augment(input_text):
    augmented_data = []

    # Swap
    aug = naw.RandomWordAug(action='swap', aug_p=config.SWAP_P)
    augmented_text = aug.augment(input_text)
    augmented_data.append(augmented_text)

    # Synonym replacement
    aug = naw.ContextualWordEmbsAug(
        model_path='HooshvareLab/bert-fa-base-uncased',
        action="substitute",
        aug_p=config.SYN_REPLACEMENT_P,
        device='cuda'
    )
    augmented_text = aug.augment(input_text)
    augmented_data.append(augmented_text)

    # Deletion
    aug = naw.RandomWordAug(action='swap', aug_p=config.DELETION_P)
    augmented_text = aug.augment(input_text)
    augmented_data.append(augmented_text)

    # Insertion
    aug = naw.ContextualWordEmbsAug(
        model_path='HooshvareLab/bert-fa-base-uncased',
        action="insert",
        aug_p=config.INSERTION_P,
        device='cuda'
    )
    augmented_text = aug.augment(input_text)
    augmented_data.append(augmented_text)

    return augmented_data


def augment_dataset(datatype='train'):
    assert datatype in ['train','val']
    texts = []
    targets = []
    if datatype == 'train':
        filepath = config.PREPROCESS_ARMAN_TRAIN_FILE
        augment_file = config.PREPROCESS_ARMAN_TRAIN_AUGMENT_FILE

    elif datatype == 'val':
        filepath = config.PREPROCESS_ARMAN_VAL_FILE
        augment_file = config.PREPROCESS_ARMAN_VAL_AUGMENT_FILE

    with open(filepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
    print("--------------------Getting the preprocessed text from file start----------------------")
    for line in lines:
        text , label = line.split('--->')
        text = text.strip()
        label = label.strip()
        data_augment_samples = augment(text)
        for sample in tqdm(data_augment_samples):
            utils.write_text_to_file(sample[0] + '--->' + str(label)+'\n',file_path=augment_file)

    del lines
