import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Input


class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        # Block 1: border detection
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), padding='same')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(2,2))

        # Block 2: detecting shapes
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2,2))

        # Block 3: detecting textures
        self.conv3 = Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2,2))

        # Block 4: high semantic
        self.conv4 = Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.bn4 = BatchNormalization()
        self.pool4 = MaxPooling2D(pool_size=(2,2))

        # High performance connector
        self.gap = GlobalAveragePooling2D()

        # Head for task 1
        self.dense_dmg = Dense(units=64, activation='relu')
        self.drop_dmg = Dropout(0.3)
        self.out_dmg = Dense(6, activation='softmax')

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training) # BN precisa saber se está em treino!
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool4(x)
        
        x_shared = self.gap(x)

        x_dmg = self.dense_dmg(x_shared)
        x_dmg = self.drop_dmg(x_dmg, training=training)
        output1 = self.out_dmg(x_dmg)

        return output1
    

def masked_sparse_loss(y_true, y_pred):
    # creates a mask that returns 1.0 if label != -1, avoids error in cases where label = -1
    mask = tf.math.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32)

    # changes -1 to 0 temporarily so Keras doesn't return error when applying loss 
    y_true_safe = tf.maximum(y_true, 0)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred)

    # applies mask, multiply by 0 all images that had -1 in its label before
    loss_mask = loss * mask

    # loss mean, dividing loss by number of valid imagens in batch
    sum_loss = tf.reduce_sum(loss_mask)

    # avoids division by 0 error in case all images in a batch have -1 label
    valid_images = tf.maximum(tf.reduce_sum(mask), 1.0)

    return sum_loss / valid_images