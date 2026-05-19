import tensorflow as tf
from src import config

# Load image from the file, decodes, resize and normalize
def load_img_info(path, label):
    img_bytes = tf.io.read_file(path) # Load
    img = tf.image.decode_jpeg(img_bytes, channels=config.CHANNELS) # Decode
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH]) # Resize

    # Crop
    img = tf.image.random_crop(
        img,
        size=[48, 48, 3]
    )

    # Image augmentation

    # Horizontal flip
    img = tf.image.random_flip_left_right(img)

    # Random brightness
    img = tf.image.random_brightness(img, max_delta=0.2)

    #Random contrast
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # Image rotation (0, 90, 180, 270)
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32
    )
    img = tf.image.rot90(img, k=k)

    img = img / 255.0  # Normalization

    # Unpack pipeline labels
    label_dmg = label[0]
    label_severity = label[1]

    return img, (label_dmg, label_severity) # returns x, y