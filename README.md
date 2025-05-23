# Road Anomaly Detection

A real-time road anomaly detection system using YOLOv8, Flask, and a modern web interface. This project detects potholes, cracks, and other road anomalies from live camera feeds, making it suitable for smart city and safety applications.

## Features

- Real-time object detection using YOLOv8
- Detects multiple road anomaly classes (vehicles, damage, bumps, pedestrians, etc.)
- Responsive web interface for desktop and mobile
- Adjustable frame rate, image quality, and resolution
- Flask backend for efficient inference and session management

## Dataset

- **Source:** [RadRoad Anomaly Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitsuresh15/radroad-anomaly-detection)
- **Classes:**
  - 0: LMV (Cars, Motorbikes, Small Trucks, Mini-vans)
  - 1: HMV (Buses, Trucks, Tractors, JCBs, Vans)
  - 2: Damage (Potholes, Cracks, Protrusions, Manholes)
  - 3: Unsurfaced (Untarred roads)
  - 4: Pedestrian
  - 5: Bump (Speed bumps)

## Model Training

- **Model:** YOLOv8s (small)
- **Epochs:** 50
- **Batch size:** 16
- **Image size:** 640x640
- **Optimizer:** AdamW
- **Data augmentation:** Enabled
- See `train.ipynb` for the full training pipeline and code

## Deployment

- **Backend:** Flask (`app.py`)
- Loads the trained YOLOv8 model for inference
- Handles image uploads, processes frames, and returns annotated resultsmost
- Multi-threaded for real-time performance

## Web Interface

- Located in `templates/index.html`
- Features live camera feed, detection overlays, and user controls
- Compatible with desktop and mobile browsers

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Flask
- OpenCV
- NumPy
- (Optional) Jupyter Notebook for training

Install dependencies:

```bash
pip install -r requirements.txt
