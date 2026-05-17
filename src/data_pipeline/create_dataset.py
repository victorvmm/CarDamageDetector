import os
import json
import tensorflow as tf
from src import config
from src.data_pipeline.transforms import load_img_info

def load_annotations(json_path):
    """Carrega as anotações do JSON mapeando o ID da imagem para a categoria."""
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r', encoding='latin-1') as f:
        data = json.load(f)
    
    mapping = {}
    for annot in data.get("annotations", []):
        # Mapeia image_id -> category_id (Dano)
        mapping[annot.get('image_id')] = annot.get('category_id')
    return mapping

def build_dataset_lists(dir_dmg, json_path, dir_sev, severity_labels_distribution):
    """
    Garante o pareamento correto entre caminhos de arquivo e rótulos [Dano, Severidade]
    """
    file_paths = []
    labels = []
    
    # 1. Processando Dataset 1 (Apenas Danos anotados no COCO/JSON)
    dmg_mapping = load_annotations(json_path)
    if os.path.exists(dir_dmg):
        for filename in os.listdir(dir_dmg):
            full_path = os.path.join(dir_dmg, filename)
            if os.path.isfile(full_path):
                # Extrai o ID numérico do arquivo (ex: "000000123.jpg" -> 123)
                try:
                    img_id = int(os.path.splitext(filename)[0])
                except ValueError:
                    continue
                
                # Se não achar no JSON, define como -1 (ausente)
                label_dano = dmg_mapping.get(img_id, -1)
                
                file_paths.append(full_path)
                # Como veio do DS1, não temos a severidade estruturada aqui -> -1
                labels.append([label_dano, -1])

    # 2. Processando Dataset 2 (Apenas Severidade organizada em subpastas)
    # severity_labels_distribution deve ser uma lista com a ordem esperada das subpastas, ex: [7, 8, 9]
    if os.path.exists(dir_sev):
        subdirs = sorted([d for d in os.listdir(dir_sev) if os.path.isdir(os.path.join(dir_sev, d))])
        
        for idx, subdir in enumerate(subdirs):
            # Garante que não estoure o índice mapeado de severidades passadas
            if idx < len(severity_labels_distribution):
                current_sev_label = severity_labels_distribution[idx]
                subdir_path = os.path.join(dir_sev, subdir)
                
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        file_paths.append(full_path)
                        # Como veio do DS2, temos severidade, mas não temos o tipo de dano -> -1
                        labels.append([-1, current_sev_label])
                        
    return file_paths, labels

def get_train_val_datasets():
    """Gera e retorna as instâncias prontas de tf.data.Dataset para Treino e Validação."""
    
    # Configuração da distribuição de severidade baseado nas subpastas sequenciais do seu DS2
    # Pasta 1 -> Severidade 7, Pasta 2 -> Severidade 8, Pasta 3 -> Severidade 9
    sev_distribution = [7, 8, 9]
    
    # Construindo listas de treino
    train_files, train_labels = build_dataset_lists(
        config.DIR_TRAIN_DMG, config.JSON_TRAIN_PATH, config.DIR_TRAIN_SEV, sev_distribution
    )
    
    # Construindo listas de validação
    val_files, val_labels = build_dataset_lists(
        config.DIR_VAL_DMG, config.JSON_VAL_PATH, config.DIR_VAL_SEV, sev_distribution
    )
    
    # Criando os Tensores com tf.data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.map(load_img_info, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=config.BUFFER_SIZE)
    train_dataset = train_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_dataset = val_dataset.map(load_img_info, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset