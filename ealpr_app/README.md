# Egyptian License Plate Recognition (EALPR)

A full-stack web application built with Flask, YOLOv8, and PyTorch for detecting and recognizing Egyptian vehicle license plates.

## Features
- **YOLOv8** for License Plate Detection
- **PyTorch CNN** for Arabic Character Recognition
- Full image preprocessing pipeline
- Modern, responsive, bilingual UI (Arabic & English)
- REST API communicating via Base64 JSON
- Automated cleanup of uploaded files

## File Structure
- `app.py`: Flask application and API routes
- `pipeline.py`: ALPR pipeline linking Detection and Recognition
- `preprocessing.py`: Image preprocessing utilities
- `models/`: Directory to store trained `.pt` and `.pth` models
- `static/`: CSS and JS assets
- `templates/`: HTML templates

## Requirements
```bash
pip install flask flask-cors ultralytics torch torchvision opencv-python numpy Pillow
```

## Setup & Running
1. Clone or extract the repository
2. Ensure the inference models are available at `models/yolo_plate.pt` and `../models/efficientnet_b0_master.pth`
3. Run the Flask server:
```bash
python app.py
```
5. Open your browser and navigate to `http://localhost:5000`

## Note
If the trained models are missing, the pipeline will load in a mock-mode for demonstration purposes, rendering a dummy bounding box and placeholder Arabic characters so the UI can be tested.
