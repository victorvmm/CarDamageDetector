from data_pipeline import load_boxes
from src import config

load_boxes(config.JSON_TRAIN_PATH, config.JSON_LABELS_TRAIN_PATH)
load_boxes(config.JSON_VAL_PATH, config.JSON_LABELS_VAL_PATH)