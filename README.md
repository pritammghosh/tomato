# Tomato grading app

This repo now has two separate parts:

- `backend/` - FastAPI service that runs the YOLO analysis and stores uploaded files and generated report JSON
- `frontend/` - React + Vite app that lets you upload an image or use the webcam and renders the full report in the browser

The backend now uses two models together:

- `best.pt` - main detector
- `bestclassifier.pt` - secondary classifier used to refine tomato status

## Run the backend

PowerShell:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe run.py
```

Git Bash:

```bash
cd backend
../.venv/Scripts/python.exe -m pip install -r requirements.txt
../.venv/Scripts/python.exe run.py
```

## Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in the browser.

If you need to point the React app at another API host, set `VITE_API_BASE_URL` before running `npm run dev` or `npm run build`.

## Notes

- The backend expects `best.pt` to remain in the repo root.
- The backend also loads `bestclassifier.pt` from the repo root by default.
- Uploaded images are saved in `backend/uploads/`.
- Generated report JSON files are saved in `backend/reports/`.
