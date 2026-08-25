import numpy as np

def compute_intersection_area(boxA, boxB):
    # box is (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h

def verify_and_clean_boxes(boxes, plate_w, plate_h):
    """
    Validates bounding boxes independently of plate tilt/perspective.
    Removes hallucinations like "مصر" or full plate bounding boxes.
    """
    if not boxes:
        return []
        
    valid_boxes = []
    
    # 1. Remove obvious non-characters based on Aspect Ratio and Size
    for (x, y, w, h) in boxes:
        area = w * h
        ar = w / float(h) if h > 0 else 0
        
        # Characters are rarely wider than they are tall by a large margin.
        # "مصر" or the plate frame will have a large aspect ratio (w > 1.3 * h).
        # FIX: Relaxed from 1.3 to 1.8 to allow slightly merged characters (like هـ ن)
        if ar > 1.8: 
            continue
            
        # Ignore extremely thin lines (like the separator) 
        # FIX: Relaxed from 0.10 to 0.05 to allow the digit "1" (١)
        if ar < 0.05:
            continue
            
        # Ignore tiny noise (less than 0.5% of plate area or height < 15% of plate height)
        if h < plate_h * 0.15 or area < (plate_w * plate_h) * 0.005:
            continue
            
        valid_boxes.append((x, y, w, h))
        
    if not valid_boxes:
        return []

    # 2. Remove Enclosing/Container Boxes (Hallucinations)
    # If a large box encapsulates smaller character boxes, the large box is a hallucination.
    boxes_to_keep = []
    for i, boxA in enumerate(valid_boxes):
        areaA = boxA[2] * boxA[3]
        arA = boxA[2] / float(boxA[3]) if boxA[3] > 0 else 0
        is_container = False
        
        for j, boxB in enumerate(valid_boxes):
            if i == j:
                continue
            
            areaB = boxB[2] * boxB[3]
            inter_area = compute_intersection_area(boxA, boxB)
            
            # If boxA is significantly larger and covers mostly boxB
            if areaA > 1.5 * areaB:
                # If boxA covers more than 60% of boxB, boxA is a container.
                if inter_area > 0.6 * areaB:
                    # FIX: Only wide boxes can be containers for multiple characters
                    # A single character box (ar < 0.85) shouldn't be deleted just because it contains a noise piece!
                    if arA > 0.85:
                        is_container = True
                        break
                    
        if not is_container:
            boxes_to_keep.append(boxA)

    if not boxes_to_keep:
        return []

    # 3. NMS (Non-Maximum Suppression)
    # Sort by area descending to keep the most prominent bounding box in case of overlaps
    boxes_to_keep.sort(key=lambda b: b[2]*b[3], reverse=True)
    final_boxes = []
    
    while boxes_to_keep:
        curr = boxes_to_keep.pop(0)
        final_boxes.append(curr)
        
        non_overlapping = []
        for b in boxes_to_keep:
            inter_area = compute_intersection_area(curr, b)
            area_b = b[2] * b[3]
            area_curr = curr[2] * curr[3]
            iou = inter_area / float(area_curr + area_b - inter_area)
            iom = inter_area / float(min(area_curr, area_b))
            
            # If overlap is greater than 30%, OR if one is mostly inside the other (>50%), suppress it
            if iou < 0.3 and iom < 0.5:
                non_overlapping.append(b)
        boxes_to_keep = non_overlapping

    # 4. Geometric Alignment Check (Perspective-Agnostic)
    # Characters form a line. We fit a line to the centers and remove outliers (e.g. the "EGYPT" box above them).
    if len(final_boxes) >= 4:
        # Sort by X to split into left and right halves
        sorted_by_x = sorted(final_boxes, key=lambda b: b[0])
        half = len(sorted_by_x) // 2
        left_half = sorted_by_x[:half]
        right_half = sorted_by_x[half:]
        
        # Median X and Y for left half
        left_x = np.median([b[0] + b[2]/2.0 for b in left_half])
        left_y = np.median([b[1] + b[3]/2.0 for b in left_half])
        
        # Median X and Y for right half
        right_x = np.median([b[0] + b[2]/2.0 for b in right_half])
        right_y = np.median([b[1] + b[3]/2.0 for b in right_half])
        
        # Slope and intercept of the robust line
        m = (right_y - left_y) / (right_x - left_x) if right_x != left_x else 0
        c = left_y - m * left_x
        
        aligned_boxes = []
        median_h = np.median([b[3] for b in final_boxes])
        for b in final_boxes:
            cx = b[0] + b[2]/2.0
            cy = b[1] + b[3]/2.0
            expected_y = m * cx + c
            
            # The "EGYPT" box or noise will be far from the characters' robust baseline
            # Also, the separator line is usually much taller than the average character
            if abs(cy - expected_y) < median_h * 0.6 and b[3] < median_h * 1.4:
                aligned_boxes.append(b)
        final_boxes = aligned_boxes
    elif len(final_boxes) >= 1:
        # Fallback to simple median Y if fewer than 4 boxes
        median_y = np.median([b[1] + b[3]/2.0 for b in final_boxes])
        median_h = np.median([b[3] for b in final_boxes])
        final_boxes = [b for b in final_boxes if abs((b[1] + b[3]/2.0) - median_y) < median_h * 0.7 and b[3] < median_h * 1.4]

    # 5. Sort horizontally Left-To-Right
    return sorted(final_boxes, key=lambda b: b[0])
