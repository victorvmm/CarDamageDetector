import os

# Image dimensions (HD)
IMG_HEIGHT = 720
IMG_WIDTH = 480
CHANNELS = 3

# Pipelines hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 1000 

# Data paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

DIR_TRAIN_DMG = os.path.join(DATA_DIR, "CarDD_release", "CarDD_COCO", "train", "train2017")
DIR_VAL_DMG = os.path.join(DATA_DIR, "CarDD_release", "CarDD_COCO", "val", "val2017")
DIR_TEST = os.path.join(DATA_DIR, "CarDD_release", "CarDD_COCO", "test", "test2017")

DIR_TRAIN_SEV = os.path.join(DATA_DIR, "training")
DIR_VAL_SEV = os.path.join(DATA_DIR, "validation")

JSON_TRAIN_PATH = os.path.join(DATA_DIR, "instances_train2017.json")
JSON_VAL_PATH = os.path.join(DATA_DIR, "instances_val2017.json")