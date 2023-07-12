import torch

from vocabulary import Vocabulary
from hazm import WordTokenizer
from transformers import AutoTokenizer

# Pytorch
HIDDEN_SIZE = None
NUM_LAYERS = None
LEARNING_RATE = 0.003
BATCH_SIZE = 256
NUM_CLASSES = 7
MAX_SEQ_LEN = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Lightning hyperparameters
MAX_EPOCHS = None
EARLY_STOPPING_PATIENCE = None
TRAIN_PERCENT_CHECK = None

# CLASS_WEIGHT:
ANGER_W = None
FEAR_W = None
HAPPINESS_W = None
WONDER_W = None
HATRESS_W = None
SADNESS_W = None

# Addition
NUMBER_OF_AUGMENTATION_WANTED = 10


# Path
PARS_EMO = 'datasets/Persian-Emotion-Detection-main/dataset.csv'
ARMAN_TRAIN = 'datasets/arman-text-emotion-main/train.tsv'
ARMAN_VAL = 'datasets/arman-text-emotion-main/test.tsv'
TEST_FILE = 'datasets/test.csv'
TAGGER_MODEL_PATH = 'resources/pos_tagger.model'
VOCAB_PATH = 'resources/vocab.pkl'
VOCAB_THRESHOLD3_PATH = 'resources/vocab.pkl'
LABELS_PATH = 'resources/labels.pkl'
CHECKPOINT_DIR = 'checkpoints/'
LOGFILE = 'logs/logfile.txt'
PREPROCESS_ARMAN_TRAIN_FILE = 'preprocessed_dataset/arman_train.txt'
PREPROCESS_ARMAN_VAL_FILE = 'preprocessed_dataset/arman_val.txt'
PREPROCESS_CONTEST_TEST_FILE = 'preprocessed_dataset/arman_train.txt'
PREPROCESS_ARMAN_TRAIN_AUGMENT_FILE = 'preprocessed_dataset/arman_train_augmented.txt'
PREPROCESS_ARMAN_VAL_AUGMENT_FILE = 'preprocessed_dataset/arman_val_augmented.txt'

# Augmentation
SWAP_P = 0.6
SYN_REPLACEMENT_P = 0.6
DELETION_P = 0.3
INSERTION_P = 0.3


# Specific functions
TOKENIZER = AutoTokenizer.from_pretrained('HooshvareLab/bert-base-parsbert-uncased')
# VOCABULARY = 