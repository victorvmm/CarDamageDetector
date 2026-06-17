import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D


class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

    def build(self, input_shape):
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

        super(MyModel, self).build(input_shape)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
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
    