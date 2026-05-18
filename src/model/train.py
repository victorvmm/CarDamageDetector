import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from src.data_pipeline import get_train_val_datasets
from src import config
from src.model.architecture import MyModel, masked_sparse_loss

train_dataset, val_dataset = get_train_val_datasets()

model = MyModel(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

def train_model(model, train_dataset, val_dataset, learning_rate, trainable=False):
    if trainable == True:
            model.base_model.trainable = True
            num_layers = len(model.base_model.layers)
            for layer in model.base_model.layers[:num_layers-30]:
                layer.trainable = False   

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'out1': masked_sparse_loss, 'out2': masked_sparse_loss},
        metrics=['accuracy', 'accuracy']
    )
    history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, validation_steps=30, verbose=2)
    return history

history1 = train_model(model, train_dataset, val_dataset, 1e-3, False)
history_fine = train_model(model, train_dataset, val_dataset, 1e-5, True)
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