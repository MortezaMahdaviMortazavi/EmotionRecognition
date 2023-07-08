import torch

from vocabulary import Vocabulary
from hazm import WordTokenizer
from transformers import AutoTokenizer

# Pytorch
HIDDEN_SIZE = None
NUM_LAYERS = None
LEARNING_RATE = None
BATCH_SIZE = None
MAX_SEQ_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Lightning hyperparameters
MAX_EPOCHS = None
EARLY_STOPPING_PATIENCE = None
TRAIN_PERCENT_CHECK = None

# CLASS_WEIGHT:
ANGER = None
FEAR = None
HAPPINESS = None
WONDER = None
HATRESS = None
SADNESS = None

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


# Augmentation
SWAP_P = 0.6
SYN_REPLACEMENT_P = 0.6
DELETION_P = 0.3
INSERTION_P = 0.3


# Specific functions
TOKENIZER = AutoTokenizer.from_pretrained('HooshvareLab/bert-base-parsbert-uncased')
# VOCABULARY = 