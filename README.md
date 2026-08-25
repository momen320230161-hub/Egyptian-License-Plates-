# Egyptian License Plate Recognition

An end-to-end computer vision project for detecting and recognizing Egyptian
vehicle license plates. It combines a Flask web application, YOLOv8 plate
detection, OpenCV preprocessing, character segmentation, and a PyTorch
EfficientNet-B0 classifier for Arabic letters and digits.

## Overview

The system accepts a vehicle image through a bilingual web interface and
returns the detected plate, a visualized crop, recognized text, confidence,
and execution time. It also exposes a JSON endpoint for integration with other
applications.

## Pipeline

```text
Input image -> YOLOv8 plate detection -> plate crop and preprocessing
                  -> character segmentation -> EfficientNet-B0 classification
                  -> Arabic letters, digits, and visualization
```

Character segmentation uses a local YOLO character checkpoint when available;
otherwise it falls back to OpenCV contour processing. The classifier applies
several test-time image variants to improve robustness.

## Repository layout

| Path | Description |
| --- | --- |
| [`ealpr_app/`](ealpr_app/) | Flask application, pipeline, frontend, and runtime models |
| [`ealpr_app/app.py`](ealpr_app/app.py) | Web server and `/predict` endpoint |
| [`ealpr_app/pipeline.py`](ealpr_app/pipeline.py) | Detection, segmentation, classification, and responses |
| [`ealpr_app/preprocessing.py`](ealpr_app/preprocessing.py) | Image preprocessing utilities |
| [`ealpr_app/bbox_checker.py`](ealpr_app/bbox_checker.py) | Bounding-box cleanup and duplicate filtering |
| [`configs/`](configs/) | YOLO dataset configurations |
| [`scripts/`](scripts/) | Dataset generation, annotation, debugging, and training |
| [`phase2/`](phase2/) | Additional training scripts and notebooks |
| [`notebooks/`](notebooks/) | Experiment and training notebooks |
| [`models/`](models/) | Shared model checkpoints and class mappings |
| [`docs/`](docs/) | Reports and project documentation |
| [`pipeline_report_images/`](pipeline_report_images/) | Evaluation figures and visuals |

## Models

The application expects these inference files:

| File | Role |
| --- | --- |
| `ealpr_app/models/yolo_plate.pt` | YOLOv8 license-plate detector |
| `models/efficientnet_b0_master.pth` | EfficientNet-B0 character classifier |

The character YOLO checkpoint is optional. The pipeline searches local
`runs/detect/.../weights/best.pt` directories; training outputs are excluded
from Git because they are generated artifacts. See `find_char_yolo()` in
[`ealpr_app/pipeline.py`](ealpr_app/pipeline.py) for the supported paths.

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision
- OpenCV, Pillow, and NumPy
- Flask and Flask-CORS
- Ultralytics YOLO

Install dependencies with:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build matching your CUDA version
before installing the remaining dependencies.

## Run the application

From the repository root:

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
cd ealpr_app
python app.py
```

Open <http://localhost:5000> and upload a vehicle image. Uploads are limited
to 16 MB and removed after processing. If a model is unavailable, the server
can still start, but the affected pipeline stage may return a fallback result
or an error.

## API

### `POST /predict`

Send an image as multipart form data using the field name `file`:

```bash
curl -X POST -F "file=@path/to/vehicle.jpg" http://localhost:5000/predict
```

Successful responses include fields similar to:

```json
{
   "success": true,
   "plate_number": "س م | ١٢٣٤",
   "confidence": 0.95,
   "execution_time": 420,
   "original_image": "data:image/jpeg;base64,...",
   "plate_crop": "data:image/jpeg;base64,..."
}
```

Errors return `success: false` and an `error` message.

## Training and experiments

- [`scripts/generate_yolo_data.py`](scripts/generate_yolo_data.py) prepares YOLO data.
- [`scripts/auto_annotate.py`](scripts/auto_annotate.py) assists with annotation.
- [`scripts/train_real_yolo_chars.py`](scripts/train_real_yolo_chars.py) trains the character detector.
- [`phase2/prepare_cnn_data.py`](phase2/prepare_cnn_data.py) prepares CNN data.
- [`phase2/train_cnn.py`](phase2/train_cnn.py) trains the character classifier.
- [`notebooks/EfficientNet_Training_V2.ipynb`](notebooks/EfficientNet_Training_V2.ipynb) contains the EfficientNet workflow.

Dataset paths may need updating on a new machine. Review the relevant script
and YAML configuration before starting a training run.

## Data and generated files

Large datasets, uploaded images, training runs, caches, duplicate checkpoints,
and temporary documents are excluded by [`.gitignore`](.gitignore). This keeps
the repository manageable and avoids distributing potentially identifiable
vehicle images.

Use a dedicated dataset host or Git LFS if the datasets must be shared, and
include the required download and license information. Never commit secrets,
private images, or local environment files.

## Documentation

Reports and supporting documents are available in [`docs/`](docs/). The cleaned
project map is available in [`project_structure.md`](project_structure.md).

## Limitations

- The system targets Egyptian plate layouts represented in the training data.
- Accuracy depends on image quality, plate visibility, and available checkpoints.
- Flask's built-in server is for local development and demonstrations; use a
   production WSGI server and security configuration for deployment.
- No authentication or persistent database is included.

## License

No license has been selected yet. Add a `LICENSE` file before public
distribution, and verify that the dataset and model weights can legally be
redistributed.