import os
import json
from src import config

def load_boxes(json_path, output_dir):
    # Checks if json path exists
    if not os.path.exists(json_path):
        return {}
    
    # Checks if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='latin-1') as f:
        data = json.load(f)

    images = {}
    for image in data.get('images', []):
        images[image['id']] = (image['width'], image['height'])

    # Loop for every damage in every image in dataset 1
    for annot in data.get('annotations', []): # Fallback = []
        img_id = annot.get('image_id')
        cat_id = annot.get('category_id')-1
        x, y, w, h = annot.get('bbox')

        img_w = images[img_id][0]
        img_h = images[img_id][1]
        
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        txt_name = f"{img_id:06d}.txt"
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'a') as txt_file:
            txt_file.write(f"{cat_id} {x_center} {y_center} {w_norm} {h_norm}\n")

if __name__ == '__main__':
    load_boxes(config.JSON_TRAIN_PATH, config.JSON_LABELS_TRAIN_PATH)
    load_boxes(config.JSON_VAL_PATH, config.JSON_LABELS_VAL_PATH)