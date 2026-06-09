from pydantic import BaseModel, Field
from typing import List

class DamageDetail(BaseModel):
    bounding_box: List[int] = Field(..., description="Coordinates of the damage bounding box")
    damage_type: str = Field(..., description="Damage class")
    confidence: float = Field(..., description="Prediction confidence")

class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Uploaded file name")
    total_damages_found: int = Field(..., description='Number of damages detected in the image')
    result: List[DamageDetail] = Field(default_factory=list, description="All damage classification details")