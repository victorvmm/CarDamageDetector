from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = 'Car Damage Detector API'
    VERSION: str = '1.0.0'

    YOLO_WEIGHTS_PATH: str = 'src/api/ml/weights/best.pt'
    CNN_WEIGHTS_PATH: str = 'src/api/ml/weights/model.weights.h5'

    HOST: str = '0.0.0.0'
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file='.venv')

settings = Settings()