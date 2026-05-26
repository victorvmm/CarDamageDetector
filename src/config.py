import os

# Image dimensions 
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

BATCH_SIZE = 32

# Data paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

DIR_TRAIN_DMG = os.path.join(DATA_DIR, "raw images", "CarDD_release", "CarDD_COCO", "train", "train2017")
DIR_VAL_DMG = os.path.join(DATA_DIR, "raw images", "CarDD_release", "CarDD_COCO", "val", "val2017")

DIR_TRAIN_SEV = os.path.join(DATA_DIR, "raw images", "training")
DIR_VAL_SEV = os.path.join(DATA_DIR, "raw images", "validation")

DIR_TEST = os.path.join(DATA_DIR, "CarDD_release", "CarDD_COCO", "test", "test2017")

JSON_TRAIN_PATH = os.path.join(DATA_DIR, "instances_train2017.json")
JSON_VAL_PATH = os.path.join(DATA_DIR, "instances_val2017.json")