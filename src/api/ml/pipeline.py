from src.classifier import MyModel
from core.config import settings
import tensorflow as tf
from src import config
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO


class Pipeline:
    def __init__(self):
        self.yolo_model = YOLO(settings.YOLO_WEIGHTS_PATH)
        self.cnn_model = MyModel()
        self.cnn_model.load_weights(settings.CNN_WEIGHTS_PATH)

    def predict(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        yolo_crop = self.yolo_model(img)
        
        detected_damages = []

        for box in yolo_crop[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            crop_img = img.crop((x1,y1,x2,y2))
            crop_img = crop_img.resize((224,224))
            img_array = tf.keras.utils.img_to_array(crop_img)
            img_array = tf.expand_dims(img_array, 0) / 255.

            cnn_pred = self.cnn_model.predict(img_array)[0]
            predicted_class = np.argmax(cnn_pred)
            confidence = float(cnn_pred[predicted_class])
            detected_damages.append({
                "bounding box": [x1,y1,x2,y2],
                "damage_type": config.CLASS_LABELS[predicted_class],
                "confidence": confidence
            })

        return detected_damages