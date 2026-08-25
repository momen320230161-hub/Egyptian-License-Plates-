import os
import cv2
import glob
import shutil
import warnings
warnings.filterwarnings('ignore')

# Import our current pipeline logic
from ealpr_app.pipeline import detect_plate_yolo, segment_characters

def auto_annotate_dataset(input_folder, output_dataset_folder):
    """
    Reads all images from input_folder, extracts the plate,
    uses our current CV pipeline to find character bounding boxes,
    and saves them in YOLO format for training a robust bounding-box model.
    """
    images_dir = os.path.join(output_dataset_folder, "images")
    labels_dir = os.path.join(output_dataset_folder, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Supported image extensions
    exts = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
        
    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_paths)} images. Starting Auto-Annotation...")
    
    success_count = 0
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 1. Detect Plate using our existing YOLO plate model
        p_box, p_conf = detect_plate_yolo(img)
        if not p_box: continue
        
        # Crop the plate
        plate_crop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
        h_p, w_p = plate_crop.shape[:2]
        if h_p == 0 or w_p == 0: continue
        
        # 2. Segment characters using our robust contour pipeline
        char_boxes = segment_characters(plate_crop)
        
        # If no characters found or too few, skip this plate to maintain dataset quality
        if len(char_boxes) < 4: continue
        
        # Prepare YOLO format annotations
        # We will train a single-class YOLO model just for "Character" (Class 0)
        yolo_annotations = []
        for (cx, cy, cw, ch) in char_boxes:
            # YOLO format requires normalized center x, center y, width, height
            x_center = (cx + cw / 2.0) / w_p
            y_center = (cy + ch / 2.0) / h_p
            n_width = cw / w_p
            n_height = ch / h_p
            
            # Ensure values are within 0-1 bounds
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            n_width = max(0.0, min(1.0, n_width))
            n_height = max(0.0, min(1.0, n_height))
            
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {n_width:.6f} {n_height:.6f}")
        
        # 3. Save the results
        base_name = f"plate_{success_count:04d}"
        
        # Save image
        out_img_path = os.path.join(images_dir, f"{base_name}.jpg")
        cv2.imwrite(out_img_path, plate_crop)
        
        # Save label
        out_lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_annotations))
            
        success_count += 1
        if success_count % 10 == 0:
            print(f"Successfully auto-annotated {success_count} plates...")

    print(f"\n✅ Auto-Annotation Complete!")
    print(f"Generated {success_count} perfectly cropped plates with YOLO labels.")
    print(f"Dataset saved at: {os.path.abspath(output_dataset_folder)}")
    print("Next step: Create a YAML file and train YOLOv8 on this folder!")

if __name__ == "__main__":
    # Change this to the path where your raw car images are
    INPUT_CARS_FOLDER = "Egyptian License Plates Dataset" 
    
    # Output folder for the new YOLO character dataset
    OUTPUT_DATASET = "real_yolo_characters_dataset"
    
    auto_annotate_dataset(INPUT_CARS_FOLDER, OUTPUT_DATASET)
