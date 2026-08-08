from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Wine Quality Prediction API is running"
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_prediction():
    payload = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert isinstance(body["prediction"], float)