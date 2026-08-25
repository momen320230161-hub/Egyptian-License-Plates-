import os, cv2, glob, shutil
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ultralytics import YOLO

YOLO_MODEL_PATH = r"c:\Users\moman\Desktop\Egyptian License Plates\phase2\runs\detect\egypt_plate_detector-2\weights\best.pt"
VEHICLES_DIR = r"c:\Users\moman\Desktop\Egyptian License Plates\Egyptian License Plates Dataset\EALPR Vechicles dataset\Vehicles"
RAW_CHARS_DIR = r"c:\Users\moman\Desktop\Egyptian License Plates\phase2\extracted_chars_raw"
SORTED_CHARS_DIR = r"c:\Users\moman\Desktop\Egyptian License Plates\phase2\sorted_characters"

os.makedirs(RAW_CHARS_DIR, exist_ok=True)
os.makedirs(SORTED_CHARS_DIR, exist_ok=True)

def segment_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w/h
        area = w*h
        if h > 15 and area > 150 and 0.1 < aspect < 2.0:
            char_crop = gray[y:y+h, x:x+w]
            char_crop = cv2.resize(char_crop, (32, 32))
            chars.append((x, char_crop))
            
    chars.sort(key=lambda x: x[0])
    return [c[1] for c in chars]

print("Loading YOLO...")
model = YOLO(YOLO_MODEL_PATH)
image_paths = glob.glob(os.path.join(VEHICLES_DIR, "*.jpg"))[:200]

print(f"Extracting characters from {len(image_paths)} images...")
char_count = 0
for img_path in image_paths:
    results = model(img_path, verbose=False)
    if results[0].boxes:
        img = cv2.imread(img_path)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size > 0:
                chars = segment_characters(plate_crop)
                for c_img in chars:
                    cv2.imwrite(os.path.join(RAW_CHARS_DIR, f"char_{char_count}.png"), c_img)
                    char_count += 1

print(f"Extracted {char_count} characters. Clustering...")

char_files = glob.glob(os.path.join(RAW_CHARS_DIR, "*.png"))
if char_files:
    data = [cv2.imread(f, 0).flatten() for f in char_files]
    kmeans = MiniBatchKMeans(n_clusters=36, random_state=42, batch_size=1000).fit(data)
    
    for i, label in enumerate(kmeans.labels_):
        cluster_path = os.path.join(SORTED_CHARS_DIR, f"cluster_{label}")
        os.makedirs(cluster_path, exist_ok=True)
        shutil.copy(char_files[i], cluster_path)
    print("Done! Ready for labeling.")
