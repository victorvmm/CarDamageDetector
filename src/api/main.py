from fastapi import FastAPI
from src.api.api.endpoints.v1.routes import router

app = FastAPI(
    title="Car Damage Detector API",
    description="API for vehicle damage classification by CNN and YOLO",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

# GET route for Docker checking if system is ready
@app.get("/health")
def health_check():
    return {"status": "online", "message": "API ready to label damages"}