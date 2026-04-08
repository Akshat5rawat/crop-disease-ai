# AI-Based Crop Disease Prediction & Management System

Production-structured starter project built with:
- Python TensorFlow model training and inference
- Flask AI API for prediction serving
- Node.js + Express backend for orchestration and history
- React + Vite frontend for image upload, camera capture, and result visualization

## Project Structure

```text
crop-disease-ai/
|-- ml-model/
|   |-- train.py
|   |-- predict.py
|   |-- compare_models.py
|   |-- gradcam.py
|   `-- requirements.txt
|-- ai-api/
|   |-- app.py
|   |-- inference.py
|   `-- requirements.txt
|-- backend/
|   |-- server.js
|   |-- models/History.js
|   `-- routes/
|-- frontend/
|   |-- src/
|   `-- package.json
`-- README.md
```

## Implemented Research Upgrades

- Grad-CAM visualization script (`ml-model/gradcam.py`)
- Multiple model comparison + graph output (`ml-model/compare_models.py`)
- Training accuracy/loss graphs (`ml-model/train.py` -> `training_curves.png`)
- Live camera capture in frontend (`frontend/src/components/CameraCapture.jsx`)

## Uniqueness Features Added

- Weather context integration via Open-Meteo API (`ai-api/app.py`)
- Disease severity scoring (`ai-api/app.py` and frontend result card)

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB running locally on `mongodb://127.0.0.1:27017`

## 1) Train Model

```bash
cd ml-model
pip install -r requirements.txt
python train.py --dataset-dir dataset --epochs 12
```

Expected dataset layout:

```text
ml-model/dataset/
|-- train/
|   |-- class_a/
|   `-- class_b/
`-- val/
    |-- class_a/
    `-- class_b/
```

## 2) Start AI API

```bash
cd ai-api
pip install -r requirements.txt
python app.py
```

Runs at `http://127.0.0.1:5000`.

## 3) Start Backend

```bash
cd backend
npm install
npm start
```

Runs at `http://localhost:4000`.

## 4) Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:3000`.

## API Endpoints

### AI API
- `GET /health`
- `POST /predict` (multipart form-data: `file`, optional `lat`, `lon`)

### Backend
- `GET /api/health`
- `POST /api/upload` (multipart form-data: `file`, optional `lat`, `lon`)
- `GET /api/history?limit=10`

## Notes

- If model loading fails, confirm `ml-model/model.h5` and `ml-model/labels.json` exist.
- If backend upload fails, ensure Flask AI API is running.
- If CORS issues appear, verify backend is reachable from frontend.