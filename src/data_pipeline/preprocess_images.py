import os 
from ultralytics import YOLO
import cv2
from src import config

def process_crop_images(origin_path, final_path, weights, padding=0):

    # Load trained model
    model = YOLO(weights)

    # Checks if final path exists
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    # Goes through origin directory
    for filename in os.listdir(origin_path):
        img_path = os.path.join(origin_path, filename)
        if not os.path.isfile(img_path):
            continue
    
        # YOLO predicts 
        results = model(img_path, verbose=False)
        result = results[0]

        # Reads original file
        img = cv2.imread(img_path)

        # For each box found
        for i, box in enumerate(result.boxes):
            # Extracts coordinates as integers
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Adds padding
            if padding > 0:
                h,w,_ = img.shape
                pad_x = int((x2-x1)*padding)
                pad_y = int((y2-y1)*padding)
                y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
                x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
            
            # Crop image
            cropped_img = img[y1:y2, x1:x2]
            
            # Save as standard name
            base_name, ext = os.path.splitext(filename)
            new_name = f"{base_name}_crop{i}{ext}"
            save_path = os.path.join(final_path, new_name)
            
            cv2.imwrite(save_path, cropped_img)

if __name__ == '__main__':
    weights_path = r'C:\Users\vmm\CarDamageDetector\src\detector\weights\best.pt' 

    process_crop_images(config.DIR_TRAIN_DMG, config.DIR_TRAIN_DMG_CROP, weights_path, padding=0.05) # Crops images from training and validation dataset 1
    process_crop_images(config.DIR_VAL_DMG, config.DIR_VAL_DMG_CROP, weights_path, padding=0.05)

    subfolder_sev = ['minor', 'moderate', 'severe']
    # Crops images from training and validation dataset 2
    for folder in subfolder_sev: 
        train = os.path.join(config.DIR_TRAIN_SEV, folder)
        final_train = os.path.join(config.DIR_TRAIN_SEV_CROP, folder)
        process_crop_images(train, final_train, weights_path, padding=0.05)
        
        val = os.path.join(config.DIR_VAL_SEV, folder)
        final_val = os.path.join(config.DIR_VAL_SEV_CROP, folder)
        process_crop_images(val, final_val, weights_path, padding=0.05)