import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bean_shape.api import create_app
from bean_shape.model import export_model, predict


def fixture_model():
    x, y = make_classification(
        n_samples=120, n_features=16, n_informative=8, n_classes=3, random_state=42
    )
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(x), y)
    model = export_model(scaler, classifier, [str(i) for i in classifier.classes_])
    return x, scaler, classifier, model


def test_json_prediction_matches_sklearn():
    x, scaler, classifier, model = fixture_model()
    result = predict(model, x[0].tolist())
    expected = classifier.predict_proba(scaler.transform(x[:1]))[0]
    np.testing.assert_allclose(result["probabilities"], expected, rtol=1e-10, atol=1e-10)
    assert result["label"] == str(classifier.predict(scaler.transform(x[:1]))[0])


def test_api_validates_input_and_returns_prediction():
    x, _, _, model = fixture_model()
    client = TestClient(create_app(model))
    response = client.post("/predict", json={"features": x[0].tolist()})
    assert response.status_code == 200
    assert abs(sum(response.json()["probabilities"]) - 1) < 1e-10
    assert client.post("/predict", json={"features": [1]}).status_code == 422
    assert client.post("/predict", json={"features": ["nan"] * 16}).status_code == 422


def test_missing_model_returns_503():
    client = TestClient(create_app(None))
    assert client.post("/predict", json={"features": [0] * 16}).status_code == 503


def test_invalid_model_rejected():
    _, _, _, model = fixture_model()
    model["scale"][0] = 0
    with pytest.raises(ValueError):
        predict(model, [0] * 16)


def test_extreme_finite_values_are_rejected():
    _, _, _, model = fixture_model()
    client = TestClient(create_app(model))
    assert client.post("/predict", json={"features": [1e308] * 16}).status_code == 422


def test_committed_artifact_matches_manifest():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "model"
    manifest = json.loads((root / "training-manifest.json").read_text())
    assert (
        hashlib.sha256((root / "model.json").read_bytes()).hexdigest()
        == manifest["artifact_sha256"]
    )
    model = json.loads((root / "model.json").read_text())
    assert predict(model, model["mean"])["label"] == "SIRA"
