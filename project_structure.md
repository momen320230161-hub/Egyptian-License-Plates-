# EALPR Project Structure Guide (Cleaned & Organized)

This document reflects the current, organized state of the project.

## 🏗️ Core Application (Production)
| Folder/File | Purpose |
| :--- | :--- |
| `ealpr_app/` | Main Flask web application. |
| `ealpr_app/pipeline.py` | **The Brain.** Orchestrates detection and recognition. |
| `models/` | Production model weights (`efficientnet_b0_master.pth`, `yolo_plate.pt`). |
| `runs/` | YOLO character detection weights and training logs. |

## 🧠 Research & Training
| Folder/File | Purpose |
| :--- | :--- |
| `notebooks/` | Contains `EfficientNet_Training_V2.ipynb` (Primary CNN trainer). |
| `scripts/` | Training scripts (`train_real_yolo_chars.py`, `train_cnn_characters.py`) and data tools. |
| `configs/` | Training configuration files (`.yaml`). |

## 📊 Datasets
| Folder | Purpose |
| :--- | :--- |
| `real_yolo_characters_dataset/` | Cleaned data for YOLO character detection. |
| `Characters/` | Cropped character images for EfficientNet classification. |
| `Egyptian License Plates Dataset/` | Source dataset. |

## 📂 Documentation & Archive
| Folder | Purpose |
| :--- | :--- |
| `docs/` | Project instructions and reports. |
| `archive/` | Redundant large files, old notebooks, and logs. |
| `phase2/` | Previous phase work. |
