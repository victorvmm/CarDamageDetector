import os

# Image dimensions 
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

BATCH_SIZE = 32
BUFFER_SIZE = 2000

CLASS_LABELS = ['Dent', 'Scratch', 'Crack', 'Glass shatter', 'Lamp broken', 'Tire flat']

# Data paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

DIR_TRAIN_DMG = os.path.join(DATA_DIR, "raw_images", "CarDD_release", "CarDD_COCO", "images", "train")
DIR_VAL_DMG = os.path.join(DATA_DIR, "raw_images", "CarDD_release", "CarDD_COCO", "images", "val")

DIR_TRAIN_SEV = os.path.join(DATA_DIR, "raw_images", "training")
DIR_VAL_SEV = os.path.join(DATA_DIR, "raw_images", "validation")

DIR_TEST = os.path.join(DATA_DIR, "CarDD_release", "CarDD_COCO", "test", "test2017")

JSON_TRAIN_PATH = os.path.join(DATA_DIR, "instances_train2017.json")
JSON_VAL_PATH = os.path.join(DATA_DIR, "instances_val2017.json")

JSON_LABELS_TRAIN_PATH = os.path.join(DATA_DIR, "raw_images", "CarDD_release", "CarDD_COCO", "labels", "train")
JSON_LABELS_VAL_PATH = os.path.join(DATA_DIR, "raw_images", "CarDD_release", "CarDD_COCO", "labels", "val")

DIR_TRAIN_DMG_CROP = os.path.join(DATA_DIR, "cropped_images", "dataset1_train")
DIR_VAL_DMG_CROP = os.path.join(DATA_DIR, "cropped_images", "dataset1_val")

DIR_TRAIN_SEV_CROP = os.path.join(DATA_DIR, "cropped_images", "dataset2_train")
DIR_VAL_SEV_CROP = os.path.join(DATA_DIR, "cropped_images", "dataset2_val")