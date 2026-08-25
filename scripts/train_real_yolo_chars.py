import torch
from ultralytics import YOLO

def train():
    print(" Starting YOLOv8 Character Bounding Box Training on Combined Dataset...")
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a pretrained YOLOv8 nano model
    # Starting from base model because we have a significantly larger dataset now (5500+ images)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    # 100 epochs is good for this size, patience will stop it if it converges early
    results = model.train(
        data='real_char_detection.yaml',
        epochs=100,
        imgsz=320,
        batch=32 if device == 0 else 16, # Larger batch if GPU
        device=device,
        workers=0, # Recommended for Windows
        name='yolo_ealpr_clean_100_epochs',
        patience=25,
        save=True
    )
    
    print(f" Training Complete! The new model is saved in: {results.save_dir}")

if __name__ == '__main__':
    train()
