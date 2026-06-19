# Car Damage Detector API

High performance API to automate car damage detection, through a deep learning two-stage pipeline using FastAPI, Docker, TensorFlow and YOLO.

## About the project

This project solves the challenge of identifying and classifying damages in vehicles in real world situations (Noisy images, different angles and lighting). Instead of using a single neural network to process the full image, the system orchestrates two models (Custom CNN and YOLOv8) working in chain to guarantee a higher precision.

## System architecture (Two-Stage Pipeline)

* Stage 1: YOLOv8 model acts as the system's eye, processing the original image to identify the damage area and generate cropped images of the corresponding region.

* Stage 2: the cropped image passes through a custom CNN (built from scratch with TensorFlow), which focuses only on the vehicle bodywork texture to classify the type of damage.

* Stage 3: the prediction is shown to the user through FastAPI, combined with Pydantic for strict data validation

## Project Structure

* `src/api/`: Complete API structure (Schemas, core modules, application entry point, routes and prediction pipeline)
* `src/classifier/`: CNN architecture, training procedures and performance visualization (Loss vs. Epoch, Accuracy vs. Epoch)
* `src/data_pipeline/`: Data processing pipelines, including image cropping, data augmentation and dataset generation for each model 
* `src/detector/`: YOLOv8 configurations files and training settings
* `tests/`: Unit tests using Pytest, including mocked prediction pipeline scenarios
* `.github/workflows/`: CI/CD workflows for automated testing and build processes

## How to execute the project

The application was packed in containers to ensure reproducibility

**1. Clone repository**
git clone https://github.com/victorvmm/CarDamageDetector.git
cd CarDamageDetector

**2. Build Docker image**
docker build -f Docker/Dockerfile -t car-damage-detector .

**3. Inicie o Container**
docker run -p 8000:8000 car-damage-detector
uvicorn src.api.main:app

**4. API testing**
Access native interactive FastAPI documentation through your navigator via:
http://localhost:8000/docs


There you open the "POST" route, click on "Try it out", upload your file and execute to get the prediction

## Licenses and data sources

This project uses open source tools and academic datasets. All the credits to its related creators:

* **YOLOv8:** Detection stage uses YOLOv8 architecture from Ultralytics, distributed by AGPL-3.0 license.

@software{
  yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

* **Dataset CarDD:** Original data for both CNN and YOLO training were based on CarDD.

https://cardd-ustc.github.io/

@ARTICLE{
  CarDD, 
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng}, 
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection}, 
  year={2023}, 
  volume={24}, 
  number={7}, 
  pages={7202-7214}, 
  doi={10.1109/TITS.2023.3258480}
}

---
Developed by Victor Melo - https://www.linkedin.com/in/victor-melo-m



