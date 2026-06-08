from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = 'Car Damage Detector API'
    VERSION: str = '1.0.0'

    YOLO_WEIGHTS_PATH: str = 'src/ml/weights/best.pt'
    CNN_WEIGHTS_PATH: str = 'src/ml/weights/model.weights.h5'

    HOST: str = '0.0.0.0'
    PORT: int = 8000

    class Config:
        env_file = '.venv'

settings = Settings()