import cv2
import numpy as np

def preprocess_pipeline(image):
    """
    Phase 1: Preprocessing for Egyptian License Plates
    Includes grayscale conversion, noise reduction, and contrast enhancement.
    """
    # 1. Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Noise reduction (Gaussian Blur or Bilateral Filter)
    # Bilateral filter preserves edges better, useful for text
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 3. Histogram Equalization (Contrast enhancement)
    # Using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced
