import pandas as pd
import nlpaug.augmenter.word as naw
import config

from tqdm import tqdm

def augment():
    targets = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]
    target = targets[2]

    df = pd.read_csv(f'/content/{target.lower()}.csv', usecols=["text", target])
    df.head()

    df = df[(df[target] > 3) | (df[target] < 2)]

    df = df.replace([1], 0)
    df = df.replace([4], 1)
    df = df.replace([5], 1)

    print(f'Value counts before augmentation: ')
    print(df[target].value_counts())

    texts = df[df[target] > 0]['text'].tolist()

    augmeneted_data = {"text": [], f"{target}": []}

    for text in tqdm(texts):
        for i in range(config.NUMBER_OF_AUGMENTATION_WANTED):
            # Swap
            aug = naw.RandomWordAug(action='swap', aug_p=config.SWAP_P)
            augmented_text = aug.augment(text)
            # Synonym replacement
            aug = naw.ContextualWordEmbsAug(model_path='HooshvareLab/bert-fa-base-uncased', action="substitute", aug_p=config.SYN_REPLACEMENT_P, device=config.DEVICE)
            augmented_text = aug.augment(augmented_text)
            # Deletion
            aug = naw.RandomWordAug(action='swap', aug_p=config.DELETION_P)
            augmented_text = aug.augment(augmented_text)
            # Insertion
            aug = naw.ContextualWordEmbsAug(model_path='HooshvareLab/bert-fa-base-uncased', action="insert", aug_p=config.INSERTION_P, device=config.DEVICE)
            augmented_text = aug.augment(augmented_text)

            augmeneted_data["text"].append(augmented_text)
            augmeneted_data[target].append(1)