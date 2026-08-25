import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Configuration
OUTPUT_DIR = "YOLO_Char_Dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

CHAR_SRC = "Characters"
NUM_PLATES = 1000 # Let's start with 1000 synthetic plates
PLATE_SIZE = (400, 150)

# Class Mapping (YOLO needs integers 0-25)
CLASS_LIST = [
    '1','2','3','4','5','6','7','8','9',
    'ain','alef','ba','dal','fa','ha','jeem','lam','meem','noon','qaf','ra','sad','seen','ta','waw','ya'
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_LIST)}

def generate_plate(index):
    # 1. Create Background (Random grayish/white plate)
    bg_color = random.randint(200, 255)
    plate = np.ones((PLATE_SIZE[1], PLATE_SIZE[0], 3), dtype=np.uint8) * bg_color
    
    # 2. Add Blue Bar (Egypt Plate Style)
    blue_bar_h = random.randint(35, 45)
    plate[0:blue_bar_h, :] = [random.randint(180, 255), random.randint(100, 150), random.randint(0, 50)] # Blue-ish BGR
    
    # 3. Pick random characters
    num_digits = random.randint(3, 4)
    num_letters = random.randint(2, 3)
    
    digits = [random.choice(os.listdir(os.path.join(CHAR_SRC, d))) for d in random.sample(CLASS_LIST[:9], num_digits)]
    letters = [random.choice(os.listdir(os.path.join(CHAR_SRC, l))) for l in random.sample(CLASS_LIST[9:], num_letters)]
    
    # Track labels for YOLO
    yolo_labels = []
    
    # 4. Paste Digits (Left side)
    curr_x = random.randint(10, 30)
    for i, char_file in enumerate(digits):
        char_name = char_file.split('_')[0] # Get class name from filename
        char_path = os.path.join(CHAR_SRC, char_name, char_file)
        char_img = cv2.imread(char_path)
        if char_img is None: continue
        
        # Random Resize
        char_h = random.randint(60, 85)
        char_w = int(char_img.shape[1] * (char_h / char_img.shape[0]))
        char_img = cv2.resize(char_img, (char_w, char_h))
        
        y_pos = random.randint(blue_bar_h + 5, PLATE_SIZE[1] - char_h - 10)
        
        # Paste with simple overlay (assuming black on white)
        char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(char_gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        roi = plate[y_pos:y_pos+char_h, curr_x:curr_x+char_w]
        if roi.shape[0] != char_h or roi.shape[1] != char_w: continue
        
        plate[y_pos:y_pos+char_h, curr_x:curr_x+char_w][mask > 0] = [0, 0, 0] # Paste black char
        
        # Save Label (class_id x_center y_center width height) normalized
        x_center = (curr_x + char_w/2) / PLATE_SIZE[0]
        y_center = (y_pos + char_h/2) / PLATE_SIZE[1]
        w_norm = char_w / PLATE_SIZE[0]
        h_norm = char_h / PLATE_SIZE[1]
        yolo_labels.append(f"{CLASS_TO_ID[char_name]} {x_center} {y_center} {w_norm} {h_norm}")
        
        curr_x += char_w + random.randint(5, 15)

    # 5. Paste Letters (Right side)
    curr_x = PLATE_SIZE[0] - 50
    for i, char_file in enumerate(letters):
        char_name = char_file.split('_')[0]
        char_path = os.path.join(CHAR_SRC, char_name, char_file)
        char_img = cv2.imread(char_path)
        if char_img is None: continue
        
        char_h = random.randint(60, 85)
        char_w = int(char_img.shape[1] * (char_h / char_img.shape[0]))
        char_img = cv2.resize(char_img, (char_w, char_h))
        
        x_pos = curr_x - char_w
        y_pos = random.randint(blue_bar_h + 5, PLATE_SIZE[1] - char_h - 10)
        
        char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(char_gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        roi = plate[y_pos:y_pos+char_h, x_pos:x_pos+char_w]
        if roi.shape[0] != char_h or roi.shape[1] != char_w: continue
        
        plate[y_pos:y_pos+char_h, x_pos:x_pos+char_w][mask > 0] = [0, 0, 0]
        
        x_center = (x_pos + char_w/2) / PLATE_SIZE[0]
        y_center = (y_pos + char_h/2) / PLATE_SIZE[1]
        w_norm = char_w / PLATE_SIZE[0]
        h_norm = char_h / PLATE_SIZE[1]
        yolo_labels.append(f"{CLASS_TO_ID[char_name]} {x_center} {y_center} {w_norm} {h_norm}")
        
        curr_x = x_pos - random.randint(5, 15)

    # 6. Save Image and Labels
    cv2.imwrite(os.path.join(IMAGES_DIR, f"plate_{index}.jpg"), plate)
    with open(os.path.join(LABELS_DIR, f"plate_{index}.txt"), 'w') as f:
        f.write("\n".join(yolo_labels))

print(f"Generating {NUM_PLATES} synthetic plates for YOLO training...")
for i in range(NUM_PLATES):
    generate_plate(i)
print("DONE Synthetic Plates and Labels Ready in 'YOLO_Char_Dataset'!")
