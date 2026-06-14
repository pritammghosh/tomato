# Tomato Grading App

This project has two parts:

- `backend/` - FastAPI service that runs the YOLO models and saves reports
- `frontend/` - React + Vite app that uploads images or uses the webcam and shows the report

The app uses two models together:

- `best.pt` - main detection model
- `bestclassifier.pt` - secondary classification model

## What You Need To Install

Before running the project, install:

1. Python 3.10 or newer
2. Node.js 18 or newer
3. Git

You can get them from:

- https://www.python.org/downloads/
- https://nodejs.org/
- https://git-scm.com/downloads

## Download The Project

Clone the repository from GitHub:

```powershell
git clone <your-github-repo-url>
cd <repo-folder>
```

## Project Files You Need

Make sure these files are in the repo root:

- `best.pt`
- `bestclassifier.pt`

The backend loads both automatically from the root folder by default.

## Backend Setup

Open PowerShell in the project root and run:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe backend\run.py
```

If your system uses a different Python command, replace `python` with the correct one.

The backend runs on:

- `http://127.0.0.1:8000`

## Frontend Setup

Open a second PowerShell window in the project root and run:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on:

- `http://localhost:5173`

## How To Use The App

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:5173`.
4. Upload an image or open the webcam.
5. Run the analysis.

## How To Confirm Both Models Are Working

After analysis, the report should show:

- `Detector + Classifier` in the model status
- A `Model check` message that names `bestclassifier.pt`
- A `classifierUsedCount` greater than `0` in the saved report JSON

If the classifier is not loaded, the UI will show `Detector only`.

## Optional Configuration

If you want to use a different classifier file, set:

```powershell
$env:CLASSIFIER_MODEL_PATH = "C:\path\to\your\classifier.pt"
```

Then start the backend again.

If the frontend should talk to a different backend host, set:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
```

## Output Folders

- `backend/uploads/` stores uploaded images
- `backend/reports/` stores generated report JSON files

## Troubleshooting

- If the backend says the model file is missing, check that `best.pt` and `bestclassifier.pt` are in the repo root.
- If the webcam does not work, use a browser that supports camera access and allow camera permissions.
- If the frontend cannot reach the backend, confirm the backend is running on port `8000`.
