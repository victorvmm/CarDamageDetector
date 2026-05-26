from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='dataset.yaml', epochs=100, imgsz=640, batch=16, device=0)