import cv2
import numpy as np
from pipeline import detect_plate_yolo, segment_characters
import matplotlib.pyplot as plt
import os

img_path = r"c:\Users\moman\Desktop\Egyptian License Plates\ealpr_app\test_bbox.jpg" 
dataset_path = r"c:\Users\moman\Desktop\Egyptian License Plates\Egyptian License Plates Dataset\EALPR Vechicles dataset\Vehicles"
test_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpeg') or f.endswith('.jpg')]

img = cv2.imread(test_images[0])
plate_box, _ = detect_plate_yolo(img)

if plate_box:
    x1, y1, x2, y2 = plate_box
    plate_crop = img[y1:y2, x1:x2]
    
    boxes = segment_characters(plate_crop)
    
    out_img = plate_crop.copy()
    w_plate = plate_crop.shape[1]
    
    from pipeline import predict_character
    
    for i, (x, y, w, h, area) in enumerate(boxes):
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        char_img = plate_crop[y:y+h, x:x+w]
        if char_img.size > 0:
            is_letter = x > w_plate * 0.45
            pred, conf = predict_character(char_img, "letter" if is_letter else "digit")
            print(f"Box {i} (x={x}, letter={is_letter}): Pred={pred}, Conf={conf:.2f}")
        
    cv2.imwrite("segmentation_test.jpg", out_img)
    print(f"Found {len(boxes)} boxes. Saved to segmentation_test.jpg")
else:
    print("No plate found.")
