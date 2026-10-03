"""Load the trained models once and use the same image preparation as training."""
import io
import json
import os
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf
from tensorflow import keras

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MODEL_FILES = {
    "mobilenetv2": "mobilenetv2.keras",
    "baseline_cnn": "baseline_cnn.keras",
}
MAX_PIXELS = 20_000_000


class InvalidImage(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def prepare_image(contents: bytes) -> tuple[tf.Tensor, int, int]:
    """Return resized RGB pixels in 0-255; scaling is already inside each model."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Please choose a JPG, PNG, or WebP image.", 415)
                if getattr(image, "is_animated", False):
                    raise InvalidImage("Please choose a still image.")
                if image.width * image.height > MAX_PIXELS:
                    raise InvalidImage("Please choose an image below 20 megapixels.")
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                width, height = rgb.size
                pixels = np.asarray(rgb, dtype=np.float32)
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidImage("Please choose an image below 20 megapixels.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise InvalidImage("This file could not be read as an image. Please choose another.") from None

    resized = tf.image.resize(pixels, (224, 224))
    return resized[None, ...], width, height


class WasteModels:
    def __init__(self, folder: Path):
        categories = json.loads((folder / "class_names.json").read_text())
        if categories != CLASSES:
            raise ValueError("The category mapping does not match the trained models.")
        self.lock = threading.Lock()
        self.models = {}
        for model_id, filename in MODEL_FILES.items():
            model = keras.models.load_model(folder / filename, compile=False)
            if model.input_shape[1:] != (224, 224, 3) or model.output_shape[-1] != 6:
                raise ValueError(f"Unexpected model shape: {filename}")
            # Warm up inference before the server starts accepting requests.
            model(tf.zeros((1, 224, 224, 3)), training=False)
            self.models[model_id] = model

    def predict(self, contents: bytes, model_ids: list[str]) -> dict:
        started = time.perf_counter()
        image, width, height = prepare_image(contents)
        predictions = []
        with self.lock:
            for model_id in model_ids:
                model_started = time.perf_counter()
                scores = self.models[model_id](image, training=False).numpy()[0]
                if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1, atol=1e-5):
                    raise RuntimeError("The model returned invalid category scores.")
                best = int(np.argmax(scores))
                predictions.append({
                    "model_id": model_id,
                    "predicted_category": CLASSES[best],
                    "model_score": float(scores[best]),
                    "category_scores": {category: float(scores[i]) for i, category in enumerate(CLASSES)},
                    "inference_ms": round((time.perf_counter() - model_started) * 1000, 1),
                })
        return {
            "width": width,
            "height": height,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "predictions": predictions,
        }

