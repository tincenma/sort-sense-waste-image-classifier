"""FastAPI endpoints and optional serving of the built React application."""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .classifier import CLASSES, InvalidImage, WasteModels

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "models")))
MAX_FILE_BYTES = 10 * 1024 * 1024
MODEL_DETAILS = json.loads((ROOT / "model_details.json").read_text())
EXAMPLES = json.loads((ROOT / "examples.json").read_text())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = await run_in_threadpool(WasteModels, MODEL_DIR)
    yield
    app.state.models = None


app = FastAPI(title="SortSense waste classifier", version="1.0.0", lifespan=lifespan)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Prediction(BaseModel):
    model_id: str
    predicted_category: str
    model_score: float
    category_scores: dict[str, float]
    inference_ms: float


class PredictionResponse(BaseModel):
    filename: str
    width: int
    height: int
    total_ms: float
    predictions: list[Prediction]


@app.get("/api/health")
def health():
    return {"status": "ready", "models": list(app.state.models.models)}


@app.get("/api/info")
def info():
    return {
        "categories": CLASSES,
        "models": MODEL_DETAILS,
        "examples": [
            {"id": item["id"], "label": item["label"], "image_url": f"/api/examples/{item['id']}"}
            for item in EXAMPLES
        ],
        "max_file_mb": 10,
    }


@app.get("/api/examples/{example_id}")
def example_image(example_id: str):
    item = next((item for item in EXAMPLES if item["id"] == example_id), None)
    if item is None:
        raise HTTPException(404, "Example not found.")
    return FileResponse(ROOT / "examples" / item["filename"], media_type="image/jpeg")


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="JPG, PNG, or WebP, up to 10 MB")],
    model: Annotated[Literal["both", "mobilenetv2", "baseline_cnn"], Form()] = "both",
):
    try:
        contents = await file.read(MAX_FILE_BYTES + 1)
        if not contents:
            raise HTTPException(400, "Please choose an image first.")
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(413, "This image is too large. The limit is 10 MB.")
        model_ids = ["mobilenetv2", "baseline_cnn"] if model == "both" else [model]
        result = await run_in_threadpool(app.state.models.predict, contents, model_ids)
        return {"filename": file.filename or "image", **result}
    except InvalidImage as error:
        raise HTTPException(error.status_code, str(error)) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Image prediction failed")
        raise HTTPException(500, "The image could not be classified. Please try again.") from None
    finally:
        await file.close()


# After `npm run build`, FastAPI can serve the complete app from one address.
frontend_dist = ROOT.parent / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

