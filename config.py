# import torch

# Pytorch
HIDDEN_SIZE = None
NUM_LAYERS = None
LEARNING_RATE = None
BATCH_SIZE = None
MAX_LEN = 128
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
NUMBER_OF_AUGMENTATION_WANTED = 300


# Path
PARS_EMO = 'datasets/Persian-Emotion-Detection-main/dataset.csv'
ARMAN_TRAIN = 'datasets/arman-text-emotion-main/train.tsv'
ARMAN_VAL = 'datasets/arman-text-emotion-main/test.tsv'
TEST_FILE = 'datasets/test.csv'


# Augmentation
SWAP_P = 0.6
SYN_REPLACEMENT_P = 0.6
DELETION_P = 0.3
INSERTION_P = 0.3



# Colors
R = "\033[0;31;40m"  # RED
G = "\033[0;32;40m"  # GREEN
Y = "\033[0;33;40m"  # Yellow
B = "\033[0;34;40m"  # Blue
N = "\033[0m"  # Reset
BG = "\033[0;37;42m"  # background green