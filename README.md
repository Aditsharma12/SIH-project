# BioScan AI: Plant Health Monitor

## What This Project Is

This is a real-time **plant health analysis web app** built with Flask and OpenCV.

It takes live camera frames from the browser, sends them to the backend, analyzes leaf regions, and returns:

- Plant health percentage
- Tissue damage percentage
- Surface coverage percentage
- A status label such as `HEALTHY`, `MODERATE`, `DAMAGED`, or `No Plant Detected`

The main video panel shows a processed stream with analysis overlays (bounding box + mesh), while a stats panel updates in real time.

## What It Does Internally

### Backend (`app.py`)

- Receives frames from browser webcam uploads (`/api/upload_frame`)
- Detects leaf-like regions using ExG (Excess Green) and HSV thresholds
- Computes health/damage/coverage metrics
- Draws visual overlays on the frame
- Streams processed video at `/video_feed`
- Streams JSON stats via SSE at `/stats_feed`

### Frontend (`templates/index.html`)

- UI dashboard for live video + metrics
- Starts browser camera upload when user clicks **Start Camera Analysis**
- Sends frames to backend every ~100ms
- Listens to SSE updates and refreshes status bars continuously

## Tech Stack

- Python
- Flask
- OpenCV (`opencv-python-headless`)
- NumPy
- Gunicorn (for Render deployment)

## Project Structure

```
SIH-project/
  app.py
  render.yaml
  requirements.txt
  templates/
    index.html
```

## Local Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run app:

   ```bash
   python app.py
   ```

3. Open:

   ```
   http://127.0.0.1:5000
   ```

4. Click **Start Camera Analysis** and allow camera permission.

## Render Deployment

Deployment config is in `render.yaml`.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4`
- `CLIENT_CAMERA=true` is set for browser-camera workflow

## Notes

- This project is currently heuristic/computer-vision based (not a trained ML model).
- Detection quality depends on lighting, camera quality, and how visible green leaf pixels are.
- If a real plant is not visible, status will likely stay `No Plant Detected `.
