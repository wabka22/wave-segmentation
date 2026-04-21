WINDOW = 512
STEP = 64

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda"

SIGNAL_DIR = "data/ecs_short"
MARKUP_DIR = "data/markings"

LABEL_CHANNEL = 0
SEED = 42

TRAIN_IDS = ["0", "4", "3", "6", "9", "14", "17", "13"]
VAL_IDS = ["16"]
TEST_IDS = ["8", "1", "5"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


TARGET_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
