import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from src.data_pipeline import get_train_val_datasets
from src import config
from src.classifier.architecture import MyModel

def train_model(model, train_dataset, val_dataset, learning_rate):
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    checkpoint_path = 'src/classifier/model.weights.h5'
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,       
        save_weights_only=True,    
        monitor='val_loss',        
        mode='min',               
        verbose=1
    )

    history = model.fit(
        train_dataset, 
        epochs=30, 
        validation_data=val_dataset, 
        validation_steps=30, 
        callbacks=[ReduceLROnPlateau('val_loss', 0.7, patience=3), checkpoint],
        verbose=1
    )
    return history

def plot_model_history(history):
    metrics = history.history

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
    fig.suptitle('Classifier model training', fontsize=18, fontweight='bold')

    # Plotting accuracy x epoch
    axes[0].plot(metrics['accuracy'], label='Training', linewidth=2, color='blue')
    axes[0].plot(metrics['val_accuracy'], label='Validation', linewidth=2, linestyle='--', color='orange')
    axes[0].set_title('Accuracy x Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # Plotting loss x epoch
    axes[1].plot(metrics['loss'], label='Treino', linewidth=2, color='red')
    axes[1].plot(metrics['val_loss'], label='Validação', linewidth=2, linestyle='--', color='green')
    axes[1].set_title('Loss x Epoch', fontsize=14)
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    train_dataset, val_dataset = get_train_val_datasets()

    model = MyModel(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

    # Show model summary
    model(tf.random.normal((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)))
    model.summary()

    history = train_model(model, train_dataset, val_dataset, 3e-4)

    plot_model_history(history)