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