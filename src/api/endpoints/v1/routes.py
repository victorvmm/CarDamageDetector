from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.api.schemas.requests import PredictionResponse, DamageDetail
from src.api.ml.pipeline import Pipeline
from src import config

router = APIRouter()

pred_pipeline = Pipeline(input_shape=(1, config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

@router.post("/predict", response_model=PredictionResponse)
async def predict_damage(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The file uploaded is not an image.")
    
    image_bytes = await file.read()
    
    prediction = pred_pipeline.predict(image_bytes)
    
    details_list = [
        DamageDetail(
            bounding_box=item["bounding_box"],
            damage_type=item["damage_type"],
            confidence=item["confidence"]
        ) for item in prediction
    ]

    return PredictionResponse(
        filename=file.filename,
        total_damages_found=len(details_list),
        details=details_list
    )