import os
import json
import tensorflow as tf
from src import config
from src.data_pipeline.transforms import load_img_info
from random import shuffle

def load_annotations(json_path):
    # Loads json annotations, mapping image id to its type of damage
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r', encoding='latin-1') as f:
        data = json.load(f)
    
    mapping = {}
    for annot in data.get("annotations", []): # fallback = []
        # Maps image_id -> category_id (Damage)
        mapping[annot.get('image_id')] = annot.get('category_id') - 1
    return mapping

def build_dataset_lists(dir_dmg, json_path, dir_sev, severity_labels_distribution, oversampling_factor):
    # Secures correct pairing between file paths and labels
    file_paths = []
    labels = []
    
    # 1. Processing Dataset 1 (Damage annotations on COCO/JSON)
    dmg_mapping = load_annotations(json_path)
    if os.path.exists(dir_dmg):
        for filename in os.listdir(dir_dmg):
            full_path = os.path.join(dir_dmg, filename)
            if os.path.isfile(full_path):
                # Extracts file ID (ex: "000000123.jpg" -> 123)
                try:
                    full_id = os.path.splitext(filename)[0]
                    id = full_id.split('_')[0]
                    img_id = int(id)
                except ValueError:
                    continue
                
                # If it does not find in JSON, defines as -1 (missing)
                label_dano = dmg_mapping.get(img_id, -1)
                
                file_paths.append(full_path)
                # As it comes from dataset 1, there isn't severity -> -1
                labels.append([label_dano, -1])

    # 2. Processing Dataset 2 (Only severity, all images organized based on its directory, each directory is a class)
    # severity_labels_distribution must be a list with the expected subfolder organization, ex: [1, 2, 3]
    if os.path.exists(dir_sev):
        subdirs = sorted([d for d in os.listdir(dir_sev) if os.path.isdir(os.path.join(dir_sev, d))])
        
        for idx, subdir in enumerate(subdirs):
            # Secures that index mapping from last severities doesn't blow
            if idx < len(severity_labels_distribution):
                current_sev_label = severity_labels_distribution[idx]
                subdir_path = os.path.join(dir_sev, subdir)
                
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        # Appends the same file for oversampling_factor size, so the proportion from dataset 1 to 2 stays almost equal
                        for _ in range(oversampling_factor):
                            file_paths.append(full_path)
                            # As it comes from dataset 2, there is severity, but not the type of damage
                            labels.append([-1, current_sev_label])
                        
    return file_paths, labels

def get_train_val_datasets():
    # Generates and returns training, validation and testing tf.data.Dataset instances

    # Folder 1 -> Severity: minor (0), Folder 2 -> Severity: moderate (1), Folder 3 -> Severity: severe (2)
    sev_distribution = [0, 1, 2]
    
    # Building training lists, oversampling factor = 5 ([ds1 train] = 7000, [ds2 train] = 1300)
    train_files, train_labels = build_dataset_lists(
        config.DIR_TRAIN_DMG_CROP, config.JSON_TRAIN_PATH, config.DIR_TRAIN_SEV_CROP, sev_distribution, 5
    )

    # Better way to shuffle dataset 
    idx_train = list(range(len(train_files)))
    shuffle(idx_train)

    train_files = [train_files[i] for i in idx_train]
    train_labels = [train_labels[i] for i in idx_train]

    # Building validation lists, oversampling factor = 8 ([ds1 val] = 2000, [ds2 val] = 260)
    val_files, val_labels = build_dataset_lists(
        config.DIR_VAL_DMG_CROP, config.JSON_VAL_PATH, config.DIR_VAL_SEV_CROP, sev_distribution, 8
    )

    # Shuffling validation dataset
    idx_val = list(range(len(val_files)))
    shuffle(idx_val)

    val_files = [val_files[i] for i in idx_val]
    val_labels = [val_labels[i] for i in idx_val]
    
    # Creating tf.data tensors
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.map(load_img_info, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=config.BUFFER_SIZE)
    train_dataset = train_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_dataset = val_dataset.map(load_img_info, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.shuffle(buffer_size=config.BUFFER_SIZE)
    val_dataset = val_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

# Returns testing dataset
def get_test_dataset():
    files_paths = []
    if os.path.exists(config.DIR_TEST):
        for filename in os.listdir(config.DIR_TEST):
            full_path = os.path.join(config.DIR_TEST, filename)                
            files_paths.append(full_path)
    test_dataset = tf.data.Dataset.from_tensor_slices(files_paths)
    test_dataset = test_dataset.shuffle(buffer_size=config.BUFFER_SIZE)
    return test_dataset