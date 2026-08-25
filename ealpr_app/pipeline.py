import os
print("!!! SURGICAL CONTOURS PIPELINE LOADED !!!", flush=True)
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import traceback
import time
from bbox_checker import verify_and_clean_boxes

def find_model_path(model_name):
    possible_paths = [
        os.path.join(os.getcwd(), "models", model_name),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", model_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name)
    ]
    for p in possible_paths:
        if os.path.exists(p): return p
    return None

YOLO_MODEL_PATH = find_model_path("yolo_plate.pt")
CNN_MODEL_PATH = find_model_path("efficientnet_b0_master.pth")

try:
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH) if YOLO_MODEL_PATH else None
except: yolo_model = None

# Smart Check for the newly training YOLO Character model
def find_char_yolo():
    possible_paths = [
        os.path.join(os.getcwd(), "runs", "detect", "yolo_ealpr_clean_100_epochs", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "detect", "yolo_ealpr_clean_100_epochs", "weights", "best.pt"),
        os.path.join(os.getcwd(), "runs", "detect", "yolo_ealpr_integrated_100_epochs-2", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "detect", "yolo_ealpr_integrated_100_epochs-2", "weights", "best.pt"),
        os.path.join(os.getcwd(), "runs", "detect", "yolo_ealpr_integrated_100_epochs", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "detect", "yolo_ealpr_integrated_100_epochs", "weights", "best.pt"),
        os.path.join(os.getcwd(), "runs", "detect", "yolo_real_chars_50_epochs", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "detect", "yolo_real_chars_50_epochs", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo_real_chars_50_epochs", "weights", "best.pt"),
        os.path.join(os.getcwd(), "runs", "detect", "yolo_real_chars", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo_real_chars", "weights", "best.pt")
    ]
    for p in possible_paths:
        if os.path.exists(p): return p
    return None

CLASS_LIST = ['1','2','3','4','5','6','7','8','9','ain','alef','ba','dal','fa','ha','jeem','lam','meem','noon','qaf','ra','sad','seen','ta','waw','ya']

def get_master_model(num_classes):
    from torchvision import models
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    return model

try:
    cnn_model = get_master_model(len(CLASS_LIST))
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    cnn_model.eval()
except: cnn_model = None

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def detect_plate_yolo(image):
    if not yolo_model: return None, 0.0
    res = yolo_model(image, verbose=False)
    if not res or not res[0].boxes: return None, 0.0
    box = res[0].boxes[torch.argmax(res[0].boxes.conf)]
    return [int(x) for x in box.xyxy[0].tolist()], box.conf[0].item()

def segment_characters(plate_img):
    if plate_img is None: return []
    
    # 🌟 NEW YOLOV8 CHARACTER SEGMENTATION 🌟
    char_yolo_path = find_char_yolo()
    
    raw_yolo_boxes = []
    if char_yolo_path:
        char_model = YOLO(char_yolo_path)
        res = char_model(plate_img, verbose=False, conf=0.15, iou=0.45)
        if res and res[0].boxes:
            for box in res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # 🌟 FILTER OUT YOLO BOXES IN THE TOP HALF OF TWO-LINE PLATES 🌟
                # This prevents YOLO from falsely detecting "EGYPT" or "مصر" as characters.
                p_h, p_w = plate_img.shape[:2]
                plate_ar = p_w / float(p_h) if p_h > 0 else 0
                if plate_ar < 3.0 and (y1 + y2) / 2.0 < p_h * 0.40:
                    continue
                    
                raw_yolo_boxes.append((cx, cy, w, h))

    # ⚠️ FALLBACK: TRADITIONAL CONTOUR LOGIC ⚠️
    h_p, w_p = plate_img.shape[:2]
    
    # 🌟 NEW: DYNAMIC ROI 🌟
    # If the plate is rectangular (two lines: EGYPT/مصر on top, characters on bottom), aspect ratio is usually < 3.0
    # If the plate is long (one line), aspect ratio is > 3.0
    plate_ar = w_p / float(h_p) if h_p > 0 else 0
    if plate_ar < 3.0:
        roi_t, roi_b = int(h_p * 0.45), int(h_p * 0.95) # Skip the top text completely
    else:
        roi_t, roi_b = int(h_p * 0.20), int(h_p * 0.95) # Include full height for long plates
        
    roi = plate_img[roi_t:roi_b, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Mild contrast boost
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    
    # Robust adaptive threshold (Preserve ink but prevent merging)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # Mild close to connect broken strokes without merging nearby characters
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Relaxed height filter to 15% for loosely cropped plates
        # Allow small boxes (dots) to pass so they can be merged with character bodies.
        # Master checker will filter out standalone noise later.
        if h > 2 and w > 1:
            # EDGE FILTER: Ignore noise/borders on the extreme left or right edges (< 2% or > 98%)
            if x < w_p * 0.02 or (x + w) > w_p * 0.98:
                continue
            raw_boxes.append([x, y, w, h])
                
    if not raw_boxes: return []
    
    raw_boxes = sorted(raw_boxes, key=lambda b: b[0])
    
    # 1. Merge ONLY horizontally overlapping boxes (fixes letters with dots above/below)
    merged_boxes = []
    curr = raw_boxes[0]
    for i in range(1, len(raw_boxes)):
        nxt = raw_boxes[i]
        overlap = max(0, min(curr[0]+curr[2], nxt[0]+nxt[2]) - max(curr[0], nxt[0]))
        # Merge if they overlap horizontally, or gap is literally 1 pixel
        gap = nxt[0] - (curr[0] + curr[2])
        if overlap > 0 or gap <= 1:
            x1 = min(curr[0], nxt[0])
            y1 = min(curr[1], nxt[1])
            x2 = max(curr[0]+curr[2], nxt[0]+nxt[2])
            y2 = max(curr[1]+curr[3], nxt[1]+nxt[3])
            curr = [x1, y1, x2-x1, y2-y1]
        else:
            merged_boxes.append(curr)
            curr = nxt
    merged_boxes.append(curr)
    
    # 2. Smart Splitting & Filtering
    final_boxes = []
    for x, y, w, h in merged_boxes:
        ar = w / float(h)
        
        # (Master Checker will filter separators later)
        
        # If box is too wide, it might be touching characters or the separator.
        if ar > 1.0 or w > w_p * 0.08:
            sub_roi = binary[y:y+h, x:x+w]
            v_hist = np.sum(sub_roi, axis=0) / 255
            
            # Dynamic valley threshold (accommodates shadows bridging the characters)
            valley_thresh = max(3.0, np.mean(v_hist) * 0.6)
            
            split_points = [0]
            in_valley = False
            for i, val in enumerate(v_hist):
                if val <= valley_thresh and not in_valley:
                    in_valley = True
                elif val > valley_thresh and in_valley:
                    split_points.append(i)
                    in_valley = False
            split_points.append(w)
            
            # If we successfully found splits
            if len(split_points) > 2:
                for i in range(len(split_points) - 1):
                    sx1 = split_points[i]
                    sx2 = split_points[i+1]
                    sw = sx2 - sx1
                    if sw > 2: 
                        sub_ar = sw / float(h)
                        sub_x = x + sx1
                        is_sub_sep = (sub_ar < 0.15) and (0.40 < (sub_x/w_p) < 0.60)
                        if not is_sub_sep and sub_ar < 2.0:
                            final_boxes.append((sub_x, y + roi_t, sw, h))
            else:
                # SAFEGUARD: If splitting failed, KEEP the box! Don't throw it away.
                final_boxes.append((x, y + roi_t, w, h))
        else:
            # --- RELAXED CHARACTER FILTERING ---
            # Master Checker will do the strict geometric filtering later.
            if 0.1 < ar < 1.5: 
                if h > h_p * 0.15 and h < h_p * 0.98:
                    final_boxes.append((x, y + roi_t, w, h))
            
    # 🌟 FILTER CONTOUR BOXES: Prioritize YOLO boxes 🌟
    # If a contour box overlaps significantly with a YOLO box, the YOLO box is more accurate.
    # Discard the contour box so it doesn't suppress the perfect YOLO box during NMS.
    filtered_contour_boxes = []
    for cb in final_boxes:
        overlap = False
        area_cb = cb[2] * cb[3]
        for yb in raw_yolo_boxes:
            area_yb = yb[2] * yb[3]
            xA = max(cb[0], yb[0])
            yA = max(cb[1], yb[1])
            xB = min(cb[0] + cb[2], yb[0] + yb[2])
            yB = min(cb[1] + cb[3], yb[1] + yb[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            
            iom = inter_area / float(min(area_cb, area_yb)) if min(area_cb, area_yb) > 0 else 0
            if iom > 0.3:
                overlap = True
                break
        if not overlap:
            filtered_contour_boxes.append(cb)
            
    # COMBINE RAW YOLO AND CONTOUR BOXES
    all_raw_boxes = raw_yolo_boxes + filtered_contour_boxes
    # 🌟 RUN THE MASTER CHECKER ON THE COMBINED SET 🌟
    # This automatically destroys container hallucinations (e.g. "٥٨" YOLO box containing two contour boxes),
    # removes overlapping duplicates, and ensures perfect horizontal alignment.
    perfect_boxes = verify_and_clean_boxes(all_raw_boxes, w_p, h_p)
    
    return perfect_boxes

def predict_character(char_img, force_type=None):
    if not cnn_model: return "?", 0.0
    from PIL import Image, ImageFilter, ImageOps
    
    # 1. Grayscale
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Dynamic Gamma Correction (Fixes Glare/Washout)
    mean_brightness = np.mean(gray)
    if mean_brightness > 160: # If too bright (glare)
        gamma = 1.5 
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
    
    # 3. CLAHE (Balanced to avoid over-sharpening shadows)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 4. Generous White Padding (Maintain Aspect Ratio & prevent edge cutoff)
    # This is CRITICAL for letters like 'Alef' and 'Dal'
    h_orig, w_orig = gray.shape
    side = int(max(w_orig, h_orig) * 1.3) # Add 30% extra space
    pad_img = np.ones((side, side), dtype=np.uint8) * 255
    x_off = (side - w_orig) // 2
    y_off = (side - h_orig) // 2
    pad_img[y_off:y_off+h_orig, x_off:x_off+w_orig] = gray
    
    # 5. Local Contrast Enhancement (Makes ink stand out in blurry images)
    gray = cv2.detailEnhance(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB), sigma_s=10, sigma_r=0.15)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    pad_img = np.ones((side, side), dtype=np.uint8) * 255
    pad_img[y_off:y_off+h_orig, x_off:x_off+w_orig] = gray
    pil = Image.fromarray(pad_img)
    
    # TTA (Test-Time Augmentation) with 4 variants for maximum robustness
    results = []
    confidences = []
    
    # Variant 1: Original
    v1 = ImageOps.autocontrast(pil).convert("RGB")
    
    # Variant 2: Sharpened
    v2 = pil.filter(ImageFilter.SHARPEN).convert("RGB")
    
    # Variant 3: Zoomed-in (Center Crop)
    w_p, h_p = pil.size
    v3 = pil.crop((w_p*0.05, h_p*0.05, w_p*0.95, h_p*0.95)).resize((w_p, h_p), Image.LANCZOS).convert("RGB")
    
    # Variant 4: High Contrast
    v4 = ImageOps.autocontrast(pil, cutoff=3).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for variant in [v1, v2, v3, v4]:
            tensor = transform(variant).unsqueeze(0)
            outputs = cnn_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            if force_type == "letter":
                idxs = [i for i, c in enumerate(CLASS_LIST) if not c.isdigit()]
            elif force_type == "digit":
                idxs = [i for i, c in enumerate(CLASS_LIST) if c.isdigit()]
            else:
                idxs = list(range(len(CLASS_LIST)))
            
            best_idx = idxs[torch.argmax(probs[idxs])]
            results.append(best_idx)
            confidences.append(probs[best_idx].item())
            
    # Hard Voting: Pick the most frequent prediction among the 3 variants
    from collections import Counter
    final_idx = Counter(results).most_common(1)[0][0]
    avg_conf = np.mean(confidences)
        
    char = CLASS_LIST[final_idx]
    arabic_map = {'alef':'أ','ba':'ب','jeem':'ج','dal':'د','ra':'ر','seen':'س','sad':'ص','ta':'ط','ain':'ع','fa':'ف','qaf':'ق','lam':'ل','meem':'م','noon':'ن','ha':'ه','waw':'و','ya':'ي','1':'١','2':'٢','3':'٣','4':'٤','5':'٥','6':'٦','7':'٧','8':'٨','9':'٩'}
    return arabic_map.get(char, char), avg_conf

def run_alpr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return {"success": False, "error": "No image"}
        start = time.time()
        p_box, p_conf = detect_plate_yolo(img)
        if not p_box: return {"success": False, "error": "No plate"}
        crop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
        cw_p = crop.shape[1]
        
        char_boxes = segment_characters(crop)
        digits, letters = [], []
        bboxes = []
        for cx, cy, cw, ch in char_boxes:
            c_img = crop[cy:cy+ch, cx:cx+cw]
            is_l = (cx > cw_p * 0.45) # Letters on right side
            pred, conf = predict_character(c_img, "letter" if is_l else "digit")
            area = cw * ch
            if is_l: letters.append((cx, pred, area))
            else: digits.append((cx, pred, area))
            bboxes.append([cx, cy, cx+cw, cy+ch])
            
        digits = sorted(digits, key=lambda x: x[2], reverse=True)[:4]
        letters = sorted(letters, key=lambda x: x[2], reverse=True)[:3]
        
        digits = sorted(digits, key=lambda x: x[0])
        letters = sorted(letters, key=lambda x: x[0], reverse=True)
        
        res_text = f'{" ".join([l[1] for l in letters])} | {"".join([d[1] for d in digits])}'
        
        out_img = img.copy()
        cv2.rectangle(out_img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 255, 0), 2)
        plate_vis = crop.copy()
        for b in bboxes: cv2.rectangle(plate_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        
        return {"success": True, "plate_number": res_text, "confidence": round(p_conf, 4), "original_image": f"data:image/jpeg;base64,{image_to_base64(out_img)}", "plate_crop": f"data:image/jpeg;base64,{image_to_base64(plate_vis)}", "execution_time": int((time.time() - start)*1000)}
    except Exception as e: return {"success": False, "error": str(e)}
