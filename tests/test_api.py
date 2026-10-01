from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "MLOps API is running"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "feature1": 5.1,
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "feature1": "wrong",
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2
        }
    )

    assert response.status_code == 422
    
    