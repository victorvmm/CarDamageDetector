from fastapi.testclient import TestClient
from unittest.mock import patch
import io
from PIL import Image
from fastapi import status

from src.api.main import app

client = TestClient(app)

# Method that creates a dummy image used for tests
def create_dummy_image():
    image = Image.new("RGB", (800,600), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Tests app health
def test_health():
    response = client.get('/health')
    assert response.status_code == status.HTTP_200_OK
    assert "status" in response.json()

# Tests how response behaves when post an invalid file type
def test_predict_invalid_file_type():
    files = {"file": ("PDF file", b"This is a false PDF", "application/pdf")}
    response = client.post('/api/v1/predict', files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "image" in response.json()["detail"].lower()

# Tests if model can predict an damage successfully
@patch("src.api.ml.pipeline.Pipeline.predict")
def test_predict_success(mock_process_image):
    mock_process_image.return_value = [
        {
            "bounding_box": [10, 20, 150, 200],
            "damage_type": "Dent",
            "confidence": 0.95
        }
    ]
    
    dummy_image = create_dummy_image()
    files = {"file": ("car_test.jpg", dummy_image, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    assert data["filename"] == "car_test.jpg"
    assert data["total_damages_found"] == 1
    assert data["result"][0]["damage_type"] == "Dent"

# Tests if model can predict an image with no damage
@patch('src.api.ml.pipeline.Pipeline.predict')
def test_predict_no_damage(mock_process_image):
    mock_process_image.return_value = []

    dummy_img = create_dummy_image()
    files = {"file": ("carro perfeito.jpg", dummy_img, "image/jpeg")}
    response = client.post('/api/v1/predict', files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["total_damages_found"] == 0
    assert len(data['result']) == 0