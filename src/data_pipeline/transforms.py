import tensorflow as tf
from src import config

# Load image from the file, decodes, resize and normalize
def load_img_info(path, label):
    img_bytes = tf.io.read_file(path) # Load
    img = tf.image.decode_jpeg(img_bytes, channels=config.CHANNELS) # Decode
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH]) # Resize
    img = img / 255.0  # Normalization

    # Unpachk pipeline labels
    label_dmg = label[0]
    label_severity = label[1]

    return img, (label_dmg, label_severity)