import cv2
import numpy as np
import os
import sys

# Append path to import from pipeline
sys.path.append(r"c:\Users\moman\Desktop\Egyptian License Plates\ealpr_app")
from pipeline import detect_plate_yolo, segment_characters

img_path = r"C:\Users\moman\.gemini\antigravity\brain\f4ac2722-4656-4c2d-9cb8-d3bd6897eddc\uploaded_media_1778087032906.img" 
img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load image from {img_path}")
    sys.exit(1)

plate_box, _ = detect_plate_yolo(img)

if plate_box:
    x1, y1, x2, y2 = plate_box
    plate_crop = img[y1:y2, x1:x2]
    
    # Save the crop to see what YOLO extracted
    cv2.imwrite("debug_yolo_crop.jpg", plate_crop)
    
    boxes = segment_characters(plate_crop)
    
    out_img = plate_crop.copy()
    for i, (x, y, w, h, area) in enumerate(boxes):
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
    cv2.imwrite("debug_segmentation.jpg", out_img)
    print(f"Found {len(boxes)} boxes. Saved to debug_segmentation.jpg")
else:
    print("No plate found by YOLO.")
