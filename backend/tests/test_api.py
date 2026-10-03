"""Exercise the real model files and the image-upload boundary."""
import io
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.classifier import prepare_image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_real_models_classify_example(client):
    image = Path(__file__).resolve().parents[1] / "examples/cardboard.jpg"
    response = client.post("/api/predict", files={"file": (image.name, image.read_bytes(), "image/jpeg")})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert {item["model_id"] for item in predictions} == {"baseline_cnn", "mobilenetv2"}
    for prediction in predictions:
        scores = prediction["category_scores"]
        assert len(scores) == 6
        assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores.values())
        assert math.isclose(sum(scores.values()), 1, abs_tol=1e-5)
        assert prediction["predicted_category"] == max(scores, key=scores.get)
    mobile = next(item for item in predictions if item["model_id"] == "mobilenetv2")
    assert mobile["predicted_category"] == "cardboard"
    assert mobile["model_score"] > 0.99


def test_single_model_and_metadata(client):
    info = client.get("/api/info").json()
    assert len(info["categories"]) == len(info["examples"]) == 6
    assert client.get("/api/health").json()["status"] == "ready"
    contents = client.get(info["examples"][0]["image_url"]).content
    result = client.post("/api/predict", data={"model": "baseline_cnn"}, files={"file": ("example.jpg", contents)})
    assert [item["model_id"] for item in result.json()["predictions"]] == ["baseline_cnn"]
    assert client.get("/api/examples/not-an-example").status_code == 404


@pytest.mark.parametrize("contents,filename,status", [
    (b"", "empty.jpg", 400),
    (b"not an image", "fake.jpg", 400),
    (b"a" * (10 * 1024 * 1024 + 1), "large.jpg", 413),
], ids=["empty", "corrupt", "too-large"])
def test_invalid_uploads(client, contents, filename, status):
    response = client.post("/api/predict", files={"file": (filename, contents, "image/jpeg")})
    assert response.status_code == status
    assert isinstance(response.json()["detail"], str)


def test_unsupported_image_format(client):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 30), "green").save(buffer, format="GIF")
    response = client.post("/api/predict", files={"file": ("image.gif", buffer.getvalue(), "image/gif")})
    assert response.status_code == 415


def test_orientation_and_unscaled_pixels():
    buffer = io.BytesIO()
    image = Image.new("RGB", (20, 30), (255, 128, 64))
    exif = Image.Exif()
    exif[274] = 6  # A phone image rotated 90 degrees clockwise.
    image.save(buffer, format="PNG", exif=exif)
    batch, width, height = prepare_image(buffer.getvalue())
    assert (width, height) == (30, 20)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.numpy()[0, 0, 0].tolist() == [255, 128, 64]
