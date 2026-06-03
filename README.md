# CarDamageDetector
Car damage detector model - Multi-task learning CNN

# Descrição
CNN (Multi-task e transfer learning):

Rede neural com backbone compartilhado que se divide em múltiplas heads para realizar tarefas simultaneas a partir de uma unica imagem

Tarefa 1: Qual o tipo de dano? (C)

Tarefa 2: Qual a severidade do dano? (C)

Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as duas cabeças da rede retornam valores no range esperado. Criar arquivo .github/workflows

API de alta performance: deploy com FastAPI e Gunicorn, usando Pydantic para validar imagem enviada não corrompida

Dockerizar tudo - criar DockerFile

# Técnicas usadas
MTL - soft parameter sharing
Data augmentation
Valor sentinela
Loss function customizada
Oversampling no dataset 2

# Abordagem

Usando soft parameter sharing

Abordagem 1: transfer learning com EfficientNetV2B3 - 27% de accuracy da saída 1 e 16% da saída 2
Underfitting, modelo muito complexo para um dataset relativamente pequeno, motivo dos erros. (+-13M de parâmetros)
1ª etapa: transfer learning clássico - rodar com modelo base congelado
2ª etapa: fine tuning com descongelamento das 30 camadas finais do modelo base com taxa de aprendizado 0.00001 (para não quebrar os pesos da 1ª etapa)

Abordagem 2: CNN criada do zero - 36.7% de accuracy da saída 1 e 18.7% da saída 2
Utiliza quatro blocos convolucionais, compostos por uma camada convolucional, uma de normalização, uma função de ativação e uma camada de maxpooling
Os blocos são conectados, por fim, às heads das tarefas por meio do GlobalAVGpool

Abordagem 3:
treino 1 - padding 5% e learning rate = 0.005 27% saídas 1 e 2
treino 2 - padding 20% e learning rate = 0.0003 30% saídas 1 e 2
treino 3 - mudanças feitas no 2 + alteração na métrica para masked accuracy

def masked_sparse_loss(y_true, y_pred):
    # CORREÇÃO CRÍTICA: Garante que y_true seja 1D (Batch_Size,) evitando bug de broadcasting
    y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])

    # Cria a máscara para ignorar os alvos -1
    mask = tf.math.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32)

    # Altera temporariamente -1 para 0 para evitar quebras no Keras
    y_true_safe = tf.where(y_true == -1, 0, y_true)

    # Calculando a loss usando from_logits=True para maior estabilidade numérica
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred, from_logits=True)

    # Aplica a máscara e calcula a média real
    loss_mask = loss * mask
    sum_loss = tf.reduce_sum(loss_mask)
    valid_images = tf.maximum(tf.reduce_sum(mask), 1.0)

    return sum_loss / valid_images


# NOVA MÉTRICA: Para você ver a acurácia real das imagens válidas no terminal
class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        mask = tf.math.not_equal(y_true, -1)
        
        # Como mudamos o modelo para Logits, pegamos o índice de maior valor
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred_labels = tf.reshape(y_pred_labels, [-1])
        
        # Verifica acertos
        correct_matches = tf.math.equal(y_true, y_pred_labels)
        correct_matches = tf.cast(correct_matches, tf.float32)
        mask_float = tf.cast(mask, tf.float32)
        
        # Acumula apenas onde a máscara for válida
        self.correct.assign_add(tf.reduce_sum(correct_matches * mask_float))
        self.total.assign_add(tf.reduce_sum(mask_float))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), # LR menor sugerida
    loss={
        'out1': masked_sparse_loss,
        'out2': masked_sparse_loss
    },
    metrics={
        'out1': [MaskedSparseCategoricalAccuracy(name='acc')],
        'out2': [MaskedSparseCategoricalAccuracy(name='acc')]
    }
)

# Head for task 1 (Dano)
self.dense_dmg = Dense(units=64, activation='relu')
self.drop_dmg = Dropout(0.3)
self.out_dmg = Dense(6, activation=None) # Mudado para None (Logits)

# Head for task 2 (Severidade)
self.dense_sev = Dense(units=64, activation='relu')
self.drop_sev = Dropout(0.3)
self.out_sev = Dense(3, activation=None) # Mudado para None (Logits)

# A FAZER


Criar notebook para data viz - imagens

Criar API

Criar dockerfile

Criar testes de modelo, API e dados

# Aprender

Docker
FastAPI
Pydantic + Gunicorn
Github actions

# Fonte

https://cardd-ustc.github.io/ - CarDD (Tarefa 1) - Pegar licença

https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data?select=data3a - Tarefa 2


# YOLO 

@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

# Pacotes

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
