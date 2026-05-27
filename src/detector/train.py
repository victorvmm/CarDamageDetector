from ultralytics import YOLO

def train_detector():

    model = YOLO('yolov8n.pt')
    results = model.train(
        data="dataset.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        device=0, 
        workers=2
    )

if __name__ == '__main__':
    train_detector()