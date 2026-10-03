# SortSense — Waste image classifier

A React and TypeScript interface with a FastAPI backend that runs our trained Small CNN and MobileNetV2 models locally. Upload an image or select one of six real TrashNet examples, then compare predictions and all category scores.

## Run the app on Windows

The working project already has its Python environment, frontend dependencies and built frontend installed. From PowerShell in this `webapp` folder:

```powershell
& ./.venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Keep the terminal running. Press **Ctrl+C** to stop the server. API documentation is at **http://127.0.0.1:8000/docs**.

## Set up the downloaded package on another computer

Install **Python 3.12** and **Node.js 22.12 or newer**. This project was tested with Python 3.12.14 and Node.js 22.12.0. No GPU or Google Colab connection is needed for inference.

Extract the ZIP, open PowerShell in its `webapp` folder, and run:

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r backend/requirements-lock.txt
& ./.venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The ZIP includes the built React interface, both model files and the example images. Open **http://127.0.0.1:8000** after the server reports that startup is complete. Initial installation downloads TensorFlow and can take several minutes.

On macOS or Linux, use `python3.12 -m venv .venv`, then `.venv/bin/python` in place of the Windows Python path above.

## Edit the frontend

Install dependencies and rebuild after changing React code:

```powershell
cd frontend
npm ci
npm run build
```

Restart FastAPI after rebuilding. For live frontend development, start the backend as above and run `npm run dev` from `frontend` in a second terminal. Open **http://127.0.0.1:5173**. Vite forwards `/api` requests to the backend on port 8000.

## What the app does

1. Accepts a JPG, PNG or WebP photo, up to 10 MB and 20 megapixels.
2. Corrects camera orientation, converts to RGB and resizes to 224 × 224 using TensorFlow bilinear resizing.
3. Sends raw floating-point pixels in the 0–255 range to each model. Pixel scaling is already embedded in the exported models.
4. Runs inference with `training=False` and returns six softmax scores.
5. Displays the highest-scoring category, every category score, model agreement and measured inference time.

Class order: **cardboard, glass, metal, paper, plastic, trash**. The app does not save uploaded photos. The six example JPEGs are unchanged images from the saved test split; their source paths are recorded in `backend/examples.json`.

## Our trained models

| Model | Held-out test accuracy | Macro F1 |
|---|---:|---:|
| MobileNetV2 | 87.30% | 0.8442 |
| Small CNN | 60.32% | 0.5857 |

These figures come from our completed training run and the same 378-image test split used in the report. MobileNetV2 was selected using validation macro F1 before test evaluation. These are dataset-level results; an individual model score is not a calibrated probability of being correct. Images with several objects, unfamiliar materials or different backgrounds may be classified incorrectly.

The models are copied unchanged from `outputs/Simple_Waste_Training/simple_results/`. They are loaded once at startup with `compile=False`. The CNN includes its original 1/255 scaling; MobileNetV2 includes its original scaling to −1…1. No training or weights modification takes place in the web application.

## Project files

```text
webapp/
  backend/
    app.py                 FastAPI routes and serving of the built UI
    classifier.py          Image preparation and real model inference
    models/                Both .keras models and class_names.json
    examples/              Six TrashNet test images
    model_details.json     Recorded test metrics
    tests/test_api.py      Model and upload checks
    requirements.txt       Pinned direct dependencies
    requirements-lock.txt  Exact installed Python packages
  frontend/
    src/App.tsx            Upload, prediction and comparison interface
    src/api.ts             Requests to FastAPI
    src/types.ts           TypeScript response types
    src/styles.css         Responsive styles
    package-lock.json      Locked frontend dependencies
    dist/                  Ready-to-serve frontend build
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness and loaded model names |
| `GET /api/info` | Categories, model test metrics and example URLs |
| `GET /api/examples/{category}` | Download a bundled example photo |
| `POST /api/predict` | Multipart upload: `file` and optional `model` |

`model` accepts `both` (default), `mobilenetv2` or `baseline_cnn`. The response contains original oriented image dimensions, total inference time and predictions with category scores for each requested model.

Optional configuration: `MODEL_DIR` points to a compatible model folder; `CORS_ORIGINS` is a comma-separated list of allowed frontend origins. A separate frontend host can use `VITE_API_BASE_URL` at build time. The default setup serves everything from one local address.

## Verification

From `webapp`:

```powershell
& ./.venv/Scripts/python.exe -m pytest backend/tests -q
cd frontend
npm run build
```

The seven API tests use the actual model files. They check predictions, all six scores, the category mapping, single-model inference, corrupt and oversized uploads, unsupported formats, orientation correction and unscaled pixels. The build checks strict TypeScript types and creates the production assets.

## References

- Dataset and bundled example source: [TrashNet repository](https://github.com/garythung/trashnet).
- [FastAPI file upload documentation](https://fastapi.tiangolo.com/tutorial/request-files/).
- [Keras model loading documentation](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/).
- [React with TypeScript](https://react.dev/learn/typescript) and [Vite](https://vite.dev/guide/).
