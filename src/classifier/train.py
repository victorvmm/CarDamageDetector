import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from src.data_pipeline import get_train_val_datasets
from src import config
from src.classifier.architecture import MyModel, masked_sparse_loss

train_dataset, val_dataset = get_train_val_datasets()

model = MyModel(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

model(tf.random.normal((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)))
model.summary()

def train_model(model, train_dataset, val_dataset, learning_rate):
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'out1': masked_sparse_loss, 'out2': masked_sparse_loss},
        loss_weights={'out1': 1.0, 'out2': 1.5},
        metrics={'out1': 'accuracy', 'out2': 'accuracy'}
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
        epochs=20, 
        validation_data=val_dataset, 
        validation_steps=30, 
        callbacks=[ReduceLROnPlateau('val_loss', 0.8), checkpoint],
        verbose=1
    )
    return history

def plot_mtl_history(history):
    """
    Plota as curvas de Acurácia e Loss para um modelo Multi-Task Learning (MTL)
    com duas saídas: 'out1' e 'out2'.
    """
    # O Keras guarda as métricas neste dicionário
    metrics = history.history

    # Criando uma figura com 2 linhas e 2 colunas (Painel 2x2)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    fig.suptitle('Diagnóstico de Treinamento: Modelo Multi-Task', fontsize=18, fontweight='bold')

    # ==========================================
    # LINHA 1: GRÁFICOS DE ACURÁCIA
    # ==========================================
    
    # 1. Acurácia - Tarefa 1 (out1)
    axes[0, 0].plot(metrics['out1_accuracy'], label='Treino', linewidth=2, color='blue')
    axes[0, 0].plot(metrics['val_out1_accuracy'], label='Validação', linewidth=2, linestyle='--', color='orange')
    axes[0, 0].set_title('Acurácia: Tarefa 1 (Dano)', fontsize=14)
    axes[0, 0].set_ylabel('Acurácia')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    # 2. Acurácia - Tarefa 2 (out2)
    axes[0, 1].plot(metrics['out2_accuracy'], label='Treino', linewidth=2, color='blue')
    axes[0, 1].plot(metrics['val_out2_accuracy'], label='Validação', linewidth=2, linestyle='--', color='orange')
    axes[0, 1].set_title('Acurácia: Tarefa 2 (Severidade)', fontsize=14)
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)

    # ==========================================
    # LINHA 2: GRÁFICOS DE PERDA (LOSS)
    # ==========================================
    
    # 3. Loss - Tarefa 1 (out1)
    axes[1, 0].plot(metrics['out1_loss'], label='Treino', linewidth=2, color='red')
    axes[1, 0].plot(metrics['val_out1_loss'], label='Validação', linewidth=2, linestyle='--', color='green')
    axes[1, 0].set_title('Perda (Loss): Tarefa 1', fontsize=14)
    axes[1, 0].set_ylabel('Loss (Erro)')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)

    # 4. Loss - Tarefa 2 (out2)
    axes[1, 1].plot(metrics['out2_loss'], label='Treino', linewidth=2, color='red')
    axes[1, 1].plot(metrics['val_out2_loss'], label='Validação', linewidth=2, linestyle='--', color='green')
    axes[1, 1].set_title('Perda (Loss): Tarefa 2', fontsize=14)
    axes[1, 1].set_ylabel('Loss (Erro)')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    # Ajusta o espaçamento para os títulos não se sobreporem
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    history = train_model(model, train_dataset, val_dataset, 3e-4)

    plot_mtl_history(history)