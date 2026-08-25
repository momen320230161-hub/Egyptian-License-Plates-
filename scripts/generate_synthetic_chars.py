import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 1. Configuration
OUTPUT_DIR = "Synthetic_Characters"
NUM_PER_CLASS = 500  # Number of images to generate per character
IMG_SIZE = (128, 128)

# Classes mapping (Key: Folder Name, Value: Arabic Character)
CHARS_MAP = {
    '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', 
    '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    'ain': 'ع', 'alef': 'أ', 'ba': 'ب', 'dal': 'د', 'fa': 'ف', 
    'ha': 'ه', 'jeem': 'ج', 'lam': 'ل', 'meem': 'م', 'noon': 'ن', 
    'qaf': 'ق', 'ra': 'ر', 'sad': 'ص', 'seen': 'س', 'ta': 'ط', 
    'waw': 'و', 'ya': 'ي'
}

# Common Windows Arabic Fonts (Arial is most standard)
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/calibri.ttf"
]

def generate_data():
    print(f"Starting Synthetic Data Factory in: {OUTPUT_DIR}")
    
    for char_name, char_val in CHARS_MAP.items():
        class_path = os.path.join(OUTPUT_DIR, char_name)
        os.makedirs(class_path, exist_ok=True)
        
        print(f"  - Generating {NUM_PER_CLASS} images for [{char_name}]...")
        
        for i in range(NUM_PER_CLASS):
            # Pick a random font from the list
            font_path = random.choice(FONT_PATHS)
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/arial.ttf" # Fallback

            # Create white canvas
            img = Image.new('L', IMG_SIZE, 255)
            draw = ImageDraw.Draw(img)
            
            # Randomize font size (mimic distance variation)
            font_size = random.randint(85, 115)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()

            # Randomize position (jitter)
            pos_x = random.randint(20, 40)
            pos_y = random.randint(5, 20)
            
            # Draw Character
            draw.text((pos_x, pos_y), char_val, font=font, fill=0)
            
            # Apply Random Transformations
            # 1. Random Rotation (mimic plate tilt)
            angle = random.randint(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=255)
            
            # 2. Random Motion Blur or Gaussian Blur
            if random.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
            
            # 3. Add Salt & Pepper Noise (mimic camera sensor noise)
            arr = np.array(img)
            noise = np.random.randint(0, 256, arr.shape, dtype='uint8')
            mask = np.random.choice([0, 1], size=arr.shape, p=[0.98, 0.02])
            arr[mask == 1] = noise[mask == 1]
            
            img = Image.fromarray(arr)
            
            # Save
            img.save(os.path.join(class_path, f"{char_name}_syn_{i}.png"))

    print("\nDONE! All synthetic characters generated successfully.")
    print(f"Path: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    generate_data()
