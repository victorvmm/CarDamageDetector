import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from src.data_pipeline import get_train_val_datasets
from src import config
from src.model.architecture import MyModel, masked_sparse_loss

train_dataset, val_dataset = get_train_val_datasets()

model = MyModel(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

model(tf.random.normal((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)))
model.summary()

def train_model(model, train_dataset, val_dataset, learning_rate):
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'out1': masked_sparse_loss, 'out2': masked_sparse_loss},
        loss_weights={'out1': 1.0, 'out2': 3.0},
        metrics={'out1': 'accuracy', 'out2': 'accuracy'}
    )

    history = model.fit(
        train_dataset, 
        epochs=20, 
        validation_data=val_dataset, 
        validation_steps=30, 
        callbacks=ReduceLROnPlateau(factor=0.5, patience=4), 
        verbose=1
    )
    return history

history1 = train_model(model, train_dataset, val_dataset, 1e-3)
'''
def show_train_results(model, history):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(history_dict['sparse_categorical_accuracy'])
    plt.plot(history_dict['val_sparse_categorical_accuracy'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.subplot(122)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show() 
'''