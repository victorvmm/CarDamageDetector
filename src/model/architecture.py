import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class MyModel(Model):
    def __init__(self, input_shape, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.base_model = EfficientNetV2B3(include_top=False, weights='imagenet', input_shape=input_shape)
        self.pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.out1 = Dense(6, activation='softmax')
        self.out2 = Dense(3, activation='softmax')

    def call(self, inputs, trainable=False):
        self.base_model.trainable = trainable
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output1 = self.out1(x)
        output2 = self.out2(x)
        return (output1, output2)
    

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