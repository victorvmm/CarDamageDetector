import tensorflow as tf
from src.data_pipeline import get_train_val_datasets
from src import config
from src.model.architecture import MyModel, masked_sparse_loss

train_dataset, val_dataset = get_train_val_datasets()

model = MyModel(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

