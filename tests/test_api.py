import pytest
from fastapi.testclient import TestClient

from clf.api import create_app
from clf.predict import Predictor


@pytest.fixture
def client(tiny_model_dir):
    with TestClient(create_app(Predictor.from_dir(tiny_model_dir))) as test_client:
        yield test_client


def test_given_two_texts_when_predicting_then_two_predictions(client):
    response = client.post("/v1/predict", json={"texts": ["card lost", "reset my pin"]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert set(predictions[0]) == {"label", "score"}


def test_given_65_texts_when_predicting_then_422(client):
    response = client.post("/v1/predict", json={"texts": ["hello"] * 65})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_given_empty_string_when_predicting_then_422(client):
    response = client.post("/v1/predict", json={"texts": [""]})

    assert response.status_code == 422


def test_given_overlong_text_when_predicting_then_422(client):
    response = client.post("/v1/predict", json={"texts": ["x" * 513]})

    assert response.status_code == 422


def test_given_no_texts_when_predicting_then_422(client):
    response = client.post("/v1/predict", json={"texts": []})

    assert response.status_code == 422


def test_given_running_app_when_checking_health_then_ok(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_given_missing_model_dir_when_starting_then_error_names_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="MODEL_DIR"), TestClient(create_app()):
        pass
