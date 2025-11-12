"""
Real-time Pest Detection using YOLOv8 and OpenCV

This program performs real-time pest detection from the webcam feed using YOLOv8.
Features:
- YOLO-style grid overlay across the entire video frame
- Bounding boxes around detected pests
- Pest name labels above each detection box
- Grid cell number where each pest was detected (optional)

Note: This creates an OpenCV GUI display window (not a Python/OS system window).
      The window is part of OpenCV's built-in image display module created by cv2.imshow().
      The window shows the Python logo icon by default (OpenCV uses Python runtime icon).
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time
import platform
import os
import argparse
import torch

# Raspberry Pi Detection - Auto-detect Raspberry Pi
IS_RASPBERRY_PI = platform.machine() in ['armv7l', 'armv6l', 'aarch64'] or \
                  'raspberry' in platform.uname().machine.lower() or \
                  (os.path.exists('/proc/device-tree/model') and 'raspberry' in open('/proc/device-tree/model', 'r').read().lower())

# Raspberry Pi Mode - Optimized for Pi 4 with 1GB RAM
RASPBERRY_PI_MODE = IS_RASPBERRY_PI  # Set to True to force Raspberry Pi optimizations

# Image enhancement settings - disabled by default for natural camera look
# Retinex is disabled by default (slow, causes latency)
# Enhancement disabled to keep camera feed looking natural
USE_FAST_ENHANCEMENT = False  # CLAHE-based enhancement (DISABLED for natural look)
RETINEX_ENABLED = False  # Multi-Scale Retinex (DISABLED by default - too slow, causes latency)
RETINEX_SCALES = [15, 80, 250]  # Multi-scale Retinex scales (optimized for clarity)
RETINEX_GAIN = 1.2  # Brightness gain for clearer images
RETINEX_CONTRAST = 1.1  # Contrast enhancement

# Pulsing animation settings for bounding boxes
PULSE_ENABLED = True  # Enable pulsing animation for bounding boxes
PULSE_DURATION = 1.0  # Duration of pulse animation in seconds
PULSE_CYCLES = 2  # Number of pulse cycles (brightness flashes)

# Human suppression settings - prevent misclassifying humans as pests
HUMAN_SUPPRESSION_ENABLED = True
HUMAN_DETECTION_INTERVAL = 3  # Run human detector every N frames
HUMAN_IOU_THRESHOLD = 0.08  # IoU threshold for overlap suppression
HUMAN_WEIGHT_THRESHOLD = 0.28  # Minimum HOG confidence to consider a human detection
HUMAN_SUPPRESSION_MARGIN = 0.25  # How much to expand detected human boxes (25% padding)
HUMAN_MIN_AREA_RATIO = 0.002  # Minimum area ratio to treat YOLO box as potential human

try:
    HOG_PEOPLE_DETECTOR = cv2.HOGDescriptor()
    HOG_PEOPLE_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception as hog_error:
    HOG_PEOPLE_DETECTOR = None
    HUMAN_SUPPRESSION_ENABLED = False
    print(f"Warning: Human suppression disabled ({hog_error})")

# Inference settings - optimized for Raspberry Pi
# NOTE: Camera resolution is NOT modified - we use whatever your camera provides
if RASPBERRY_PI_MODE:
    # Raspberry Pi 4 with 1GB RAM: Use smaller inference size to reduce memory
    INFERENCE_SIZE = 320  # Reduced from 640 to save memory (~4x less memory)
    # CAMERA_WIDTH and CAMERA_HEIGHT are NOT used - camera uses its own settings
    print("=" * 60)
    print("RASPBERRY PI MODE ENABLED")
    print("Using optimized settings for Pi 4 with 1GB RAM:")
    print(f"  - Inference size: {INFERENCE_SIZE}px (reduced for memory)")
    print("  - Camera: Using camera's native resolution (NOT modified)")
    print(f"  - Retinex: DISABLED (memory intensive)")
    print("=" * 60)
else:
    # Standard PC mode - Optimized for ACCURACY
    INFERENCE_SIZE = 640  # Increased from 320 for better detection accuracy
    # Larger inference size = better accuracy (more detail for detection)
    # CAMERA_WIDTH and CAMERA_HEIGHT are NOT used - camera uses its own settings

# Class names matching YOUR dataset (12 pest classes from data.yaml)
CLASS_NAMES = [
    'Ants',           # 0
    'Bees',           # 1
    'Beetles',        # 2
    'Caterpillars',   # 3
    'Earthworms',     # 4
    'Earwigs',        # 5
    'Grasshoppers',   # 6
    'Moths',          # 7
    'Slugs',          # 8
    'Snails',         # 9
    'Wasps',          # 10
    'Weevils'         # 11
]

# Color palette for different classes (BGR format for OpenCV) - 12 colors
# Exact hex color codes as specified:
# 🐜 Ants — Green #00FF00
# 🐝 Bees — Yellow #FFFF00
# 🪲 Beetles — Red #FF0000
# 🐛 Caterpillars — Orange #FFA500
# 🪱 Earthworms — Purple #800080
# 🦗 Earwigs — Silver #C0C0C0
# 🦟 Grasshoppers — Yellow #FFD700
# 🦋 Moths — Magenta #FF00FF
# 🐌 Slugs — Dark Green #006400
# 🐚 Snails — Pink #FFC0CB
# 🐝 Wasps — Blue #0000FF
# 🪶 Weevils — White #FFFFFF
def hex_to_bgr(hex_color):
    """Convert hex color to BGR format for OpenCV"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # BGR format for OpenCV

COLORS = [
    hex_to_bgr('00FF00'),  # Ants - Green #00FF00
    hex_to_bgr('FFFF00'),  # Bees - Yellow #FFFF00
    hex_to_bgr('FF0000'),  # Beetles - Red #FF0000
    hex_to_bgr('FFA500'),  # Caterpillars - Orange #FFA500
    hex_to_bgr('800080'),  # Earthworms - Purple #800080
    hex_to_bgr('C0C0C0'),  # Earwigs - Silver #C0C0C0
    hex_to_bgr('FFD700'),  # Grasshoppers - Yellow #FFD700
    hex_to_bgr('FF00FF'),  # Moths - Magenta #FF00FF
    hex_to_bgr('006400'),  # Slugs - Dark Green #006400
    hex_to_bgr('FFC0CB'),  # Snails - Pink #FFC0CB
    hex_to_bgr('0000FF'),  # Wasps - Blue #0000FF
    hex_to_bgr('FFFFFF'),  # Weevils - White #FFFFFF
]

def draw_yolo_grid(frame, grid_size=32, alpha=0.3):
    """
    Draw a YOLO-style grid overlay across the entire video frame
    Creates a dynamic grid overlay like YOLO training visualization
    
    Args:
        frame: Input video frame
        grid_size: Size of each grid cell in pixels (default: 32 for YOLO-style)
        alpha: Transparency of grid overlay (0.0-1.0, default: 0.3)
    
    Returns:
        frame with grid overlay drawn
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Create overlay for grid (allows transparency)
    overlay = frame.copy()
    
    # Draw vertical grid lines
    x = 0
    cell_col = 0
    while x < frame_width:
        # Alternate line thickness for better visibility (every 4th line is thicker)
        line_thickness = 2 if cell_col % 4 == 0 else 1
        line_color = (150, 150, 150) if cell_col % 4 == 0 else (80, 80, 80)
        cv2.line(overlay, (x, 0), (x, frame_height), line_color, line_thickness)
        x += grid_size
        cell_col += 1
    
    # Draw horizontal grid lines
    y = 0
    cell_row = 0
    while y < frame_height:
        # Alternate line thickness for better visibility (every 4th line is thicker)
        line_thickness = 2 if cell_row % 4 == 0 else 1
        line_color = (150, 150, 150) if cell_row % 4 == 0 else (80, 80, 80)
        cv2.line(overlay, (0, y), (frame_width, y), line_color, line_thickness)
        y += grid_size
        cell_row += 1
    
    # Blend overlay with original frame for transparency effect
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame

def get_grid_cell(x, y, frame_width, frame_height, grid_size=32):
    """
    Calculate which grid cell a point (x, y) belongs to
    
    Args:
        x, y: Point coordinates
        frame_width, frame_height: Frame dimensions
        grid_size: Size of each grid cell in pixels
    
    Returns:
        (grid_col, grid_row): Grid cell coordinates (0-indexed)
    """
    grid_col = min(int(x / grid_size), int(frame_width / grid_size) - 1)
    grid_row = min(int(y / grid_size), int(frame_height / grid_size) - 1)
    return grid_col, grid_row

def draw_grid_around_detection(
    frame,
    x1,
    y1,
    x2,
    y2,
    color,
    grid_size=16,
    line_thickness=2,
    expand_ratio=0.10,
    label=None,
    label_color=None,
    font_scale=0.7,
    font_thickness=2,
):
    """
    Draw a bright grid pattern around the detected pest bounding box.
    The grid is automatically expanded by a configurable ratio (default 10%)
    beyond the pest to give more visual context, and the pest label can be
    rendered within the grid region.
    
    Args:
        frame: Input video frame
        x1, y1, x2, y2: Bounding box coordinates
        color: Grid line color (BGR format)
        grid_size: Size of each grid cell in pixels (default: 16)
        line_thickness: Thickness of grid lines (default: 2 for better visibility)
        expand_ratio: Fractional padding applied to width/height (default: 0.10 = 10%)
        label: Optional text to display centered within the grid region
        label_color: Optional text color (defaults to `color` if None)
        font_scale: Font scale for label text
        font_thickness: Font thickness for label text
    
    Returns:
        frame with grid drawn inside detection
    """
    # Ensure coordinates are valid
    if x1 >= x2 or y1 >= y2:
        return frame
    
    frame_height, frame_width = frame.shape[:2]

    # Expand the grid region by the specified ratio
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * expand_ratio / 2.0)
    pad_y = int(height * expand_ratio / 2.0)

    grid_x1 = max(0, x1 - pad_x)
    grid_y1 = max(0, y1 - pad_y)
    grid_x2 = min(frame_width - 1, x2 + pad_x)
    grid_y2 = min(frame_height - 1, y2 + pad_y)

    grid_width = max(1, grid_x2 - grid_x1)
    grid_height = max(1, grid_y2 - grid_y1)

    # Slightly brighten the grid color so it stands out
    brighten_factor = 1.35
    grid_color = tuple(min(255, int(c * brighten_factor)) for c in color)
    border_color = tuple(min(255, int(c * 1.5)) for c in color)

    # Adjust grid size to roughly match pest size (keeping minimum and maximum bounds)
    adaptive_grid_size = max(6, min(int(min(grid_width, grid_height) / 8), 24))
    if grid_size:
        adaptive_grid_size = max(4, int(grid_size))
    grid_step = max(4, adaptive_grid_size)

    # Draw vertical grid lines inside the expanded region directly on the frame
    for x in range(grid_x1, grid_x2 + 1, grid_step):
        cv2.line(frame, (x, grid_y1), (x, grid_y2), grid_color, line_thickness, lineType=cv2.LINE_AA)

    # Draw horizontal grid lines inside the expanded region directly on the frame
    for y in range(grid_y1, grid_y2 + 1, grid_step):
        cv2.line(frame, (grid_x1, y), (grid_x2, y), grid_color, line_thickness, lineType=cv2.LINE_AA)

    # Draw a visible border around the expanded grid region
    cv2.rectangle(
        frame,
        (grid_x1, grid_y1),
        (grid_x2, grid_y2),
        border_color,
        max(2, line_thickness + 1),
        lineType=cv2.LINE_AA,
    )

    # Render pest label inside the grid region if provided
    if label:
        if label_color is None:
            contrast_color = tuple(255 - min(220, c) for c in grid_color)
            label_color = contrast_color
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # Center the label in the expanded grid region
        text_x = grid_x1 + max(0, (grid_width - text_width) // 2)
        center_y = grid_y1 + grid_height // 2
        text_y = int(center_y + text_height / 2)
        text_y = min(grid_y2 - baseline, max(grid_y1 + text_height, text_y))
        
        # Draw translucent background behind the label for readability
        label_bg_padding = 6
        bg_x1 = max(0, text_x - label_bg_padding)
        bg_y1 = max(0, text_y - text_height - label_bg_padding)
        bg_x2 = min(frame_width - 1, text_x + text_width + label_bg_padding)
        bg_y2 = min(frame_height - 1, text_y + label_bg_padding)
        cv2.rectangle(
            frame,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            font_scale,
            label_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    
    return frame

def box_iou(box_a, box_b):
    """Compute IoU between two boxes defined as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area + 1e-6
    
    return inter_area / union_area

def detect_humans_in_frame(frame):
    """
    Detect humans in the frame using HOG person detector to suppress false positives.
    Returns list of (x1, y1, x2, y2) boxes in original frame coordinates.
    """
    if not HUMAN_SUPPRESSION_ENABLED or HOG_PEOPLE_DETECTOR is None:
        return []
    
    if frame is None or frame.size == 0:
        return []
    
    # HOG works better on slightly larger frames; upscale modestly if frame is small
    scale_factor = 1.0
    height, width = frame.shape[:2]
    max_dimension = max(height, width)
    if max_dimension < 640:
        scale_factor = 640.0 / max_dimension
        resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    else:
        resized_frame = frame
    
    rects, weights = HOG_PEOPLE_DETECTOR.detectMultiScale(
        resized_frame,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )
    
    human_boxes = []
    for idx, rect in enumerate(rects):
        x, y, w, h = rect
        if len(weights) > idx:
            weight_value = weights[idx]
            try:
                weight = float(weight_value)
            except (TypeError, ValueError):
                weight = float(weight_value[0]) if hasattr(weight_value, '__iter__') else 1.0
        else:
            weight = 1.0
        if weight < HUMAN_WEIGHT_THRESHOLD:
            continue  # Skip weak detections
        # Convert back to original frame coordinates if scaled
        if scale_factor != 1.0:
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
        # Expand box to create safer suppression area
        expand_w = int(w * HUMAN_SUPPRESSION_MARGIN)
        expand_h = int(h * HUMAN_SUPPRESSION_MARGIN)
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h)
        x2 = min(width - 1, x + w + expand_w)
        y2 = min(height - 1, y + h + expand_h)
        human_boxes.append((x1, y1, x2, y2))
    
    return human_boxes

def draw_detections(frame, results, confidence_threshold=0.25, detection_timestamps=None, current_time=None, detected_pests=None, human_boxes=None):
    """
    Draw bounding boxes and labels on the frame with false positive filtering
    Includes pulsing animation for newly detected pests
    Draws grid around each detected pest and announces pest names
    
    Args:
        frame: Input video frame
        results: YOLOv8 detection results
        confidence_threshold: Minimum confidence to display detection
        detection_timestamps: Dictionary tracking when each pest was first detected (for pulsing animation)
        current_time: Current time in seconds (for pulsing animation)
        detected_pests: Set to track which pests have been announced (prevents spam)
        human_boxes: List of human detections to suppress overlapping pest boxes
    """
    if detection_timestamps is None:
        detection_timestamps = {}
    if current_time is None:
        current_time = time.time()
    if detected_pests is None:
        detected_pests = set()
    if human_boxes is None:
        human_boxes = []
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    
    for result in results:
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Only process if confidence is above threshold
            if confidence < confidence_threshold:
                continue  # Skip low confidence detections
            
            # Validate class_id - ensure it's within valid range for our pest classes
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue  # Skip invalid class IDs (not in our pest list)
            
            # Calculate bounding box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Expand bounding box by 10% (1.1x total size) - 5% padding on each side
            # This makes the box 10% bigger than the insect (1.1x size)
            expansion_factor = 1.1  # 10% expansion (1.1x size)
            padding_x = box_width * (expansion_factor - 1.0) / 2.0  # 5% padding on left/right
            padding_y = box_height * (expansion_factor - 1.0) / 2.0  # 5% padding on top/bottom
            
            # Calculate expanded bounding box coordinates
            expanded_x1 = max(0, int(x1 - padding_x))
            expanded_y1 = max(0, int(y1 - padding_y))
            expanded_x2 = min(frame_width, int(x2 + padding_x))
            expanded_y2 = min(frame_height, int(y2 + padding_y))
            
            # Use expanded coordinates for drawing
            draw_x1, draw_y1, draw_x2, draw_y2 = expanded_x1, expanded_y1, expanded_x2, expanded_y2
            
            # Calculate area for filtering (use original detection size for filtering)
            box_area = box_width * box_height
            aspect_ratio = box_height / (box_width + 1e-6)  # Avoid division by zero
            area_ratio = box_area / frame_area  # Ratio of box area to frame area
            
            # HUMAN SUPPRESSION: Skip detections overlapping detected humans
            if human_boxes:
                skip_detection = False
                pest_box = (x1, y1, x2, y2)
                for human_box in human_boxes:
                    if box_iou(pest_box, human_box) > HUMAN_IOU_THRESHOLD:
                        skip_detection = True
                        break
                    hx1, hy1, hx2, hy2 = human_box
                    if hx1 <= center_x <= hx2 and hy1 <= center_y <= hy2:
                        skip_detection = True
                        break
                if skip_detection:
                    continue  # Overlaps human detection - suppress to avoid false positives
            else:
                # Even without explicit human boxes, guard against human-like regions
                if area_ratio >= HUMAN_MIN_AREA_RATIO and (aspect_ratio > 1.3 or box_height > 80 or box_width > 80):
                    continue  # Likely part of a human silhouette
                if area_ratio >= HUMAN_MIN_AREA_RATIO and (x1 <= 4 or y1 <= 4 or x2 >= frame_width - 4 or y2 >= frame_height - 4):
                    continue  # Large object stretching to frame edge
            
            # ULTRA-STRICT PEST-ONLY FILTERING
            # Only detect actual pests - reject humans and other large objects
            # Insects are TINY, square-ish or slightly rectangular
            # Humans are LARGE, TALL and THIN - NOT IN PEST DATASET
            
            MAX_AREA_RATIO = 0.006  # Maximum 0.6% of frame area (insects are VERY small)
            MIN_AREA_RATIO = 0.00005  # Minimum 0.005% of frame area (very lenient for tiny pests)
            MAX_ABSOLUTE_HEIGHT = 50  # Maximum 50 pixels tall (insects are TINY)
            MAX_ABSOLUTE_WIDTH = 50  # Maximum 50 pixels wide (insects are TINY)
            MAX_ABSOLUTE_AREA = MAX_ABSOLUTE_WIDTH * MAX_ABSOLUTE_HEIGHT  # Cap for large blobs
            
            # Get class name for reference
            class_name = CLASS_NAMES[class_id]
            
            # PRE-FILTER #1: Reject very large detections immediately (likely humans)
            # If detection is > 2% of frame, it's definitely not an insect
            if area_ratio > 0.015:
                continue  # Skip silently - too large to be an insect
            
            # PRE-FILTER #2: Reject if width or height > 10% of frame (definitely human)
            if box_width > 0.10 * frame_width or box_height > 0.10 * frame_height:
                continue  # Skip silently - too large to be an insect
            
            # PRE-FILTER #3: Reject ANY large detection regardless of confidence
            # Large objects are always humans, even with high confidence
            if area_ratio > 0.01:
                continue  # Skip silently - too large, likely human
            
            # PRE-FILTER #4: Reject low confidence + any significant size
            # Large objects with low confidence are usually misclassified humans
            if confidence < 0.7 and area_ratio > 0.006:
                continue  # Skip silently - low confidence + large = false positive
            
            # PRE-FILTER #5: Reject if absolute pixel size > 60px
            # Insects are tiny - they're never more than 60 pixels
            if box_width > 54 or box_height > 54:
                continue  # Skip silently - too large in pixels
            
            # PRE-FILTER #6: Reject if area still very large in absolute pixels
            if box_area > MAX_ABSOLUTE_AREA:
                continue  # Skip - absolute area too large
            
            # PRE-FILTER #7: Reject detections hugging frame borders with significant size (likely human)
            EDGE_MARGIN = 32  # pixels
            if (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > frame_width - EDGE_MARGIN or y2 > frame_height - EDGE_MARGIN) and area_ratio > 0.002:
                continue  # Skip - large object near frame edge, likely human
            
            # FILTER #1: Reject ANY tall objects (aspect ratio > 1.2) - likely human body
            # Insects can be slightly elongated but not tall
            if aspect_ratio > 1.2:
                continue  # Too tall - not an insect
            
            # FILTER #2: Reject objects larger than 1% of frame area
            # Insects are tiny - they never take up more than 1% of frame
            if area_ratio > MAX_AREA_RATIO:
                continue  # Too large - not an insect
            
            # FILTER #3: Reject objects larger than 10% of frame width/height
            # Insects are tiny - check if width or height is more than 10% of frame
            if box_width > 0.10 * frame_width or box_height > 0.10 * frame_height:
                continue  # Too large relative to frame - not an insect
            
            # FILTER #4: Reject objects with absolute pixel size > 60px
            # Insects are tiny - they're never more than 60 pixels in any dimension
            if box_width > MAX_ABSOLUTE_WIDTH or box_height > MAX_ABSOLUTE_HEIGHT:
                continue  # Too large in pixels - not an insect
            
            # FILTER #5: Reject tall rectangles (height > 1.2x width)
            # Insects are not tall - if height > 1.2x width, it's likely human
            if box_height > box_width * 1.2:
                continue  # Too tall rectangle - likely human
            
            # FILTER #6: Reject moderately sized objects with tall aspect ratio
            # Objects that are tall and moderately sized are likely humans
            if aspect_ratio > 1.1 and area_ratio > 0.008:
                continue  # Tall and moderately sized - likely human
            
            # FILTER #7: Reject wide but large objects (human torso)
            if aspect_ratio < 0.7 and area_ratio > 0.01:
                continue  # Wide and large - likely human torso
            
            # FILTER #8: Reject objects that are both large and oddly shaped
            # Insects are small and have reasonable proportions
            if (box_width > 0.07 * frame_width or box_height > 0.07 * frame_height) and \
               (aspect_ratio > 1.3 or aspect_ratio < 0.7):
                continue  # Large and oddly shaped - likely human
            
            # FILTER #9: Reject any object that's both large in area AND tall
            # Insects are never both large and tall
            if area_ratio > 0.0065 and aspect_ratio > 1.1:
                continue  # Large area and tall - definitely human
            
            # FILTER #10: Reject anything with aspect ratio outside 0.7-1.2
            # Insects have reasonable proportions - not too tall or too wide
            if aspect_ratio < 0.7 or aspect_ratio > 1.2:
                continue  # Unusual proportions - likely human
            
            # FILTER #11: Reject if width or height > 8% of frame (catch-all)
            # Insects are never that large
            if box_width > 0.07 * frame_width or box_height > 0.07 * frame_height:
                continue  # Too large in any dimension - likely human
            
            # Passed all filters - this is a real pest!
            
            # Filter out extremely tiny detections (likely noise)
            if area_ratio < MIN_AREA_RATIO:
                continue  # Skip - too small, likely noise
            
            # Passed all filters - draw the detection
            color = COLORS[class_id]
            
            # Histogram equalization DISABLED for natural camera look
            # Enhancement disabled to keep camera feed looking natural
            # Extract region around pest with padding (disabled enhancement)
            # padding = 20  # Padding around bounding box (disabled)
            # roi_x1 = max(0, x1 - padding)
            # roi_y1 = max(0, y1 - padding)
            # roi_x2 = min(frame_width, x2 + padding)
            # roi_y2 = min(frame_height, y2 + padding)
            # 
            # # Extract region of interest (ROI) around the pest
            # roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            # Enhancement disabled - no histogram equalization applied for natural camera look
            # All image enhancement code removed to keep camera feed looking natural
            
            # Create unique detection ID for tracking pulsing animation
            detection_id = f"{class_id}_{x1}_{y1}_{x2}_{y2}"
            
            # Track detection timestamp for pulsing animation
            if detection_id not in detection_timestamps:
                detection_timestamps[detection_id] = current_time
            
            # Calculate pulsing animation intensity
            pulse_intensity = 1.0
            if PULSE_ENABLED:
                time_since_detection = current_time - detection_timestamps[detection_id]
                if time_since_detection < PULSE_DURATION:
                    # Create pulsing effect using sine wave
                    pulse_phase = (time_since_detection / PULSE_DURATION) * PULSE_CYCLES * 2 * np.pi
                    pulse_intensity = 0.5 + 0.5 * (1 + np.sin(pulse_phase))  # Oscillates between 0.5 and 1.0
                else:
                    # Remove old detections from tracking
                    if time_since_detection > PULSE_DURATION * 2:
                        detection_timestamps.pop(detection_id, None)
            
            # Apply pulsing effect to color (brighten during pulse)
            if PULSE_ENABLED and pulse_intensity > 1.0:
                # Brighten the color during pulse
                pulse_color = tuple(min(255, int(c * pulse_intensity)) for c in color)
            else:
                pulse_color = color
            
            # Draw YOLO-style bounding box - clean, thick colored box
            # Standard YOLO visualization: thick colored rectangle with label above
            
            # Calculate box thickness with pulsing effect
            base_thickness = 3  # Standard YOLO box thickness
            box_thickness = int(base_thickness * pulse_intensity) if PULSE_ENABLED else base_thickness
            box_thickness = max(2, min(box_thickness, 6))  # Keep thickness reasonable (2-6px)
            
            # Draw main bounding box (thick colored rectangle) - expanded by 10% for better visibility
            # Box is drawn 1.1x larger than the detected pest (10% expansion)
            cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), pulse_color, box_thickness)
            
            # Draw grid around the detected pest inside the bounding box
            draw_grid_around_detection(
                frame,
                draw_x1,
                draw_y1,
                draw_x2,
                draw_y2,
                pulse_color,
                grid_size=0,
                line_thickness=1,
                label=pest_name,
                label_color=pulse_color,
            )
            
            # Draw inner box for depth effect (optional, makes box more prominent)
            inner_thickness = 1
            cv2.rectangle(frame, (draw_x1+2, draw_y1+2), (draw_x2-2, draw_y2-2), pulse_color, inner_thickness)
            
            # Get pest name and announce it (print to console if new detection)
            pest_name = class_name
            # Use original detection coordinates for tracking (not expanded box)
            detection_key = f"{pest_name}_{x1}_{y1}"
            
            # Announce new pest detections (avoid spam by tracking)
            if detection_key not in detected_pests:
                print(f"🐛 PEST DETECTED: {pest_name} (Confidence: {confidence:.0%}) at ({x1}, {y1})")
                detected_pests.add(detection_key)
                # Clean old detections from set to allow re-announcement after some time
                if len(detected_pests) > 50:
                    detected_pests.clear()
            
            # Draw prominent pest name label - larger and more visible
            # Standard YOLO format: class_name confidence%
            label_text = f"{pest_name} {confidence:.0%}"  # e.g., "Ants 85%"
            
            # Calculate text size for proper label background (larger font for visibility)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9  # Increased from 0.7 for better visibility
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Label position: above top-left corner of expanded bounding box
            label_padding = 8  # Increased padding for better visibility
            label_x1 = draw_x1
            label_y1 = max(0, draw_y1 - text_height - label_padding * 2)  # Ensure label is not above frame
            label_x2 = draw_x1 + text_width + label_padding * 2
            label_y2 = draw_y1
            
            # Draw label background (colored box matching bounding box color)
            cv2.rectangle(
                frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,  # Use class color for background
                -1  # Filled rectangle
            )
            
            # Draw label text (white for contrast on colored background)
            text_x = draw_x1 + label_padding
            text_y = draw_y1 - label_padding
            cv2.putText(
                frame,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # White text for maximum contrast
                font_thickness,
                cv2.LINE_AA
            )
            
            # Draw pest name again below the expanded box for additional visibility
            bottom_label_y = min(frame_height - 10, draw_y2 + text_height + label_padding)
            bottom_label_y1 = bottom_label_y - text_height - label_padding * 2
            cv2.rectangle(
                frame,
                (draw_x1, bottom_label_y1),
                (draw_x1 + text_width + label_padding * 2, bottom_label_y),
                color,
                -1
            )
            cv2.putText(
                frame,
                pest_name,  # Just the name, no confidence on bottom
                (draw_x1 + label_padding, bottom_label_y - label_padding),
                font,
                font_scale * 0.8,  # Slightly smaller for bottom label
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
    
    return frame

def apply_fast_enhancement(frame, alpha=1.0, beta=0):
    """
    Frame Enhancement Pipeline for clear view:
    1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for brightness balance
    2. Use fastNlMeansDenoisingColored for noise removal
    3. Add sharpening kernel for detail clarity
    4. Optionally adjust alpha and beta for contrast and brightness
    
    Args:
        frame: Input frame
        alpha: Contrast control (1.0 = no change, >1.0 = more contrast)
        beta: Brightness control (0 = no change, positive = brighter)
    
    Returns:
        Enhanced frame
    """
    try:
        # OPTIMIZED for LOW LATENCY - lighter processing
        # Step 1: Skip expensive denoising for speed (causes latency)
        # Use faster bilateral filter instead or skip entirely
        # denoised = cv2.fastNlMeansDenoisingColored(frame, None, h=4, templateWindowSize=7, searchWindowSize=21)
        denoised = frame  # Skip denoising for lower latency
        
        # Step 2: Convert to LAB color space for better contrast enhancement
        # CLAHE works best on the L (lightness) channel
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for brightness balance
        # Lighter CLAHE for lower latency (lower clipLimit = faster)
        # tileGridSize: Size of grid for adaptive equalization (smaller = more localized)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Step 4: Merge enhanced channels
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 5: Lightweight sharpening (skip heavy sharpening for lower latency)
        # Use lighter sharpening or skip entirely for lower latency
        # kernel = np.array([[-0.75, -0.75, -0.75],
        #                   [-0.75,  6.0, -0.75],
        #                   [-0.75, -0.75, -0.75]])
        # sharpened = cv2.filter2D(enhanced, -1, kernel)
        # result = cv2.addWeighted(enhanced, 0.15, sharpened, 0.85, 0)
        result = enhanced  # Skip sharpening for lower latency
        
        # Step 6: Optional alpha and beta adjustment for contrast and brightness
        if alpha != 1.0 or beta != 0:
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result
    except Exception as e:
        # If enhancement fails, return original frame
        return frame

def apply_retinex(frame):
    """
    Apply Multi-Scale Retinex (MSR) algorithm for image enhancement
    Enhances image quality, especially for low-light conditions
    Makes camera detection clearer and more accurate
    NOTE: This is slower - consider using apply_fast_enhancement for better FPS
    """
    if not RETINEX_ENABLED:
        return frame
    
    try:
        # Convert BGR to float and normalize
        img = frame.astype(np.float64) / 255.0
        
        # Apply Multi-Scale Retinex with optimized scales for clear detection
        msr = np.zeros_like(img)
        
        for scale in RETINEX_SCALES:
            # Gaussian blur - kernel size calculated from sigma
            kernel_size = int(6 * scale + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), scale)
            # Avoid division by zero
            blurred = np.maximum(blurred, 1e-10)
            # Retinex calculation - log ratio
            msr += np.log(img + 1e-10) - np.log(blurred + 1e-10)
        
        # Average across scales
        msr = msr / len(RETINEX_SCALES)
        
        # Enhanced normalization for clearer output
        # Get min/max per channel for better normalization
        for c in range(3):
            channel = msr[:, :, c]
            if channel.max() > channel.min():
                msr[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)
        
        # Restore to 0-255 range
        msr = msr * 255.0
        
        # Apply brightness and contrast enhancement for clearer detection
        msr = np.clip(msr, 0, 255).astype(np.uint8)
        
        # Apply adaptive histogram equalization (CLAHE) for better clarity
        lab = cv2.cvtColor(msr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Optional: Additional light sharpening for clearer edges
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.05 + np.eye(3) * 0.95)
        
        # Blend enhanced (95%) with sharpened (5%) for clear detection
        final = cv2.addWeighted(enhanced, 0.95, sharpened, 0.05, 0)
        
        return final
    
    except Exception as e:
        # If Retinex fails, return original frame
        return frame

def main():
    """
    Main function for real-time pest detection using OpenCV
    - Uses OpenCV VideoCapture to access camera
    - Uses OpenCV imshow to display output on screen
    - Real-time detection and visualization
    - Supports command-line arguments for confidence threshold
    - Supports video saving with annotations
    - Optimized for GPU if available
    """
    global RETINEX_ENABLED, USE_FAST_ENHANCEMENT  # Allow toggling enhancement in main loop
    import os
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-time Pest Detection using YOLOv8')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection (default: 0.25, lower = more detections)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output video with annotations to output.avi')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to custom YOLOv8 model (default: auto-detect trained model or yolov8n.pt)')
    parser.add_argument('--model-size', type=str, choices=['n', 's'], default='n',
                        help='YOLOv8 model size: n (nano) or s (small) (default: n)')
    parser.add_argument('--source', type=str, default=None,
                        help='Video source: webcam index (0, 1, etc.) or path to video file (default: 0 for webcam)')
    parser.add_argument('--grid-size', type=int, default=32,
                        help='YOLO-style grid cell size in pixels (default: 32)')
    parser.add_argument('--grid-alpha', type=float, default=0.3,
                        help='Grid overlay transparency (0.0-1.0, default: 0.3)')
    args = parser.parse_args()
    
    # Check GPU availability and optimize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("Real-time Pest Detection using YOLOv8")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("GPU acceleration enabled for faster inference!")
    else:
        print("CPU mode: GPU not available, using CPU")
    print("=" * 60)
    print("Loading model and initializing camera...\n")
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Try trained model first (check multiple possible locations)
        # Find the most recent trained model from any pest_detection folder
        trained_model_paths = []
        
        # Check for the most recent training run
        if os.path.exists('runs/train'):
            # Find all pest_detection folders
            for item in os.listdir('runs/train'):
                folder_path = os.path.join('runs/train', item)
                if os.path.isdir(folder_path) and 'pest_detection' in item:
                    model_file = os.path.join(folder_path, 'weights', 'best.pt')
                    if os.path.exists(model_file):
                        trained_model_paths.append(model_file)
            
            # Sort by modification time (most recent first)
            trained_model_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Also check standard locations
        standard_paths = [
            'runs/detect/train/weights/best.pt',
            'runs/train/weights/best.pt',
        ]
        trained_model_paths.extend([p for p in standard_paths if os.path.exists(p)])
        
        model_path = None
        if trained_model_paths:
            model_path = trained_model_paths[0]  # Use most recent trained model
            print(f"✓ Found trained model: {model_path}\n")
            print("  Using YOUR trained model (trained on your dataset!)\n")
        
        if model_path is None:
            # Use pretrained YOLOv8 model
            model_path = f'yolov8{args.model_size}.pt'
            print(f"NOTE: Using pretrained YOLOv8{args.model_size} model")
            print("      Train your model for pest-specific detection: python train.py\n")
    
    # Load model
    try:
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        # Move model to GPU if available (YOLO handles this automatically, but we can verify)
        print(f"Model loaded successfully on {device.upper()}!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Initialize OpenCV VideoCapture - support both webcam and video file input
    cap = None
    is_video_file = False
    original_camera_index = 0
    original_backend = cv2.CAP_ANY
    source_found = False
    
    # Determine video source (webcam or video file)
    if args.source:
        # Check if source is a video file or webcam index
        if os.path.isfile(args.source):
            # Video file input
            print(f"\nOpening video file: {args.source}")
            cap = cv2.VideoCapture(args.source)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✓ Video file opened successfully!")
                    print(f"  Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    is_video_file = True
                    source_found = True
                else:
                    cap.release()
                    cap = None
        else:
            # Try as webcam index
            try:
                camera_index = int(args.source)
                print(f"\nOpening camera index {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"✓ Camera {camera_index} opened successfully!")
                        print(f"  Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        original_camera_index = camera_index
                        source_found = True
                    else:
                        cap.release()
                        cap = None
            except ValueError:
                print(f"\nERROR: Invalid source '{args.source}'. Must be a webcam index (0, 1, etc.) or video file path.")
                return
    else:
        # Default: try to open webcam (camera index 0)
        print("\nOpening camera with OpenCV...")
        import sys
        is_windows = sys.platform.startswith('win')
        
        # Try different backends and indices
        backends_to_try = []
        if is_windows:
            # On Windows, try DSHOW backend first
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backends_to_try = [cv2.CAP_ANY]
        
        for backend in backends_to_try:
            if source_found:
                break
            print(f"Trying backend: {backend}")
            
            for camera_index in range(5):
                print(f"  Trying camera index {camera_index}...")
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    if cap.isOpened():
                        # Test if we can actually read frames
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"✓ Camera {camera_index} opened successfully!")
                            print(f"  Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            # Store original camera settings for reconnection
                            original_camera_index = camera_index
                            original_backend = backend
                            source_found = True
                            break
                        else:
                            cap.release()
                            cap = None
                except Exception as e:
                    print(f"  Error trying camera {camera_index}: {e}")
                    if cap:
                        cap.release()
                        cap = None
        
        if not source_found or not cap or not cap.isOpened():
            print("\nERROR: Could not open camera with OpenCV!")
            print("\nPlease check:")
            print("  1. Camera is connected to your computer")
            print("  2. Camera is not in use by another application")
            print("  3. Camera drivers are installed")
            print("  4. Try closing other apps that might use the camera")
            print("  5. Or specify a video file: --source path/to/video.mp4")
            print("\nTrying one more time with default settings...")
            
            # Last attempt with simple VideoCapture
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print("✓ Camera opened on second attempt!")
                    original_camera_index = 0
                    original_backend = cv2.CAP_ANY
                    source_found = True
                else:
                    cap.release()
                    cap = None
    
    if not source_found or not cap or not cap.isOpened():
        print("\nFailed to open video source. Exiting...")
        print("Usage: python detect_realtime.py --source 0  (for webcam)")
        print("   or: python detect_realtime.py --source path/to/video.mp4  (for video file)")
        return
    
    # Get actual camera properties (use whatever settings the camera already has)
    # Do NOT modify camera settings - work with existing configuration
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nUsing camera's current settings:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps:.1f}\n")
    
    # Create OpenCV GUI display window
    # This creates an OpenCV image display window (not a Python/OS system window)
    # The window is part of OpenCV's built-in image display module created by cv2.imshow()
    # Note: Window shows Python logo icon by default (OpenCV uses Python runtime icon)
    window_name = 'Pest Detection - Real-time'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Adjust window size to match camera resolution (adapt to camera settings)
    # Use actual camera resolution for window size
    display_width = actual_width
    display_height = actual_height
    
    # Limit maximum window size for very high resolutions (optional, for better UX)
    max_display_width = 1920
    max_display_height = 1080
    if display_width > max_display_width or display_height > max_display_height:
        scale = min(max_display_width / display_width, max_display_height / display_height)
        display_width = int(display_width * scale)
        display_height = int(display_height * scale)
        print(f"  Window size adjusted to: {display_width}x{display_height} (for display)")
    
    # Set window size to match camera resolution (or scaled version)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # Configure window properties for better appearance
    try:
        # Set window properties
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except:
        pass
    
    # Show an initial frame immediately so user knows window is open
    print("\nOpening detection window...")
    ret, initial_frame = cap.read()
    if ret and initial_frame is not None:
        # Resize frame to match display window size (adapt to camera settings)
        height, width = initial_frame.shape[:2]
        if width != display_width or height != display_height:
            # Resize to match display window size
            initial_frame = cv2.resize(initial_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        
        # Add text to initial frame
        cv2.putText(initial_frame, "Camera Ready!", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(initial_frame, "Starting detection...", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the initial frame
        cv2.imshow(window_name, initial_frame)
        cv2.waitKey(100)  # Brief pause to ensure window displays
        print(f"✓ Detection window opened at {display_width}x{display_height} (adapted to camera settings)!")
    
    print("\n" + "=" * 60)
    print("Real-time Pest Detection Started!")
    print("=" * 60)
    print(f"\nDetecting 12 pest classes:")
    print(f"  {', '.join(CLASS_NAMES)}")
    print("\nFrame Enhancement Pipeline:")
    print("  ✓ CLAHE (Contrast Limited Adaptive Histogram Equalization) for brightness balance")
    print("  ✓ fastNlMeansDenoisingColored for noise removal")
    print("  ✓ Sharpening kernel for detail clarity")
    print("  ✓ Optional alpha/beta adjustment for contrast and brightness")
    print("\nCamera Settings:")
    print(f"  ✓ Using camera's current configuration")
    print(f"  ✓ Resolution: {actual_width}x{actual_height}")
    # Use command-line confidence threshold (set before printing)
    confidence_threshold = args.conf
    
    print("\nOptimizations Active:")
    if torch.cuda.is_available():
        print("  ✓ GPU acceleration enabled")
    else:
        print("  ✓ CPU mode (GPU not available)")
    print(f"  ✓ Confidence threshold: {confidence_threshold}")
    print("  ✓ Real-time FPS counter")
    print("  ✓ Detection status display")
    print("\nControls:")
    print("  Press 'q' or ESC - Quit")
    print("  Press 's' - Save current frame")
    print("  Press 'r' - Toggle Retinex on/off (CLAHE stays enabled)")
    print("  Press 'e' - Toggle all image enhancement on/off")
    print("  Press '+' - Increase confidence threshold")
    print("  Press '-' - Decrease confidence threshold")
    print("\nCamera feed is now displaying on your screen!")
    print("Detection window is visible and showing real-time output!\n")
    
    # Get actual camera resolution for video writer
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if saving is enabled
    video_writer = None
    if args.save_video:
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = 'output.avi'
        fps = 20.0  # Output FPS
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))
        if video_writer.isOpened():
            print(f"Video saving enabled: {output_path}")
            print(f"  Resolution: {actual_width}x{actual_height}, FPS: {fps}")
        else:
            print(f"Warning: Failed to open video writer for {output_path}")
            video_writer = None
    
    # FPS calculation variables
    fps_counter = 0
    fps_timer = cv2.getTickCount()
    fps_display = 0
    detection_status = "Detecting..."
    
    # Initialize detection timestamps for pulsing animation
    detection_timestamps = {}
    
    # Initialize detected pests set for tracking announcements
    detected_pests = set()
    
    # Human detection cache for suppression
    human_boxes = []
    frames_since_human_detect = HUMAN_DETECTION_INTERVAL
    
    # Start detection loop - camera stays ON until 'q' or ESC is pressed
    print("Camera is now running and will stay ON until you press 'q' or ESC.\n")
    
    running = True
    try:
        # Continuous loop - camera stays on until user presses quit key
        while running:
            try:
                # Verify camera is still open
                if not cap.isOpened():
                    print("ERROR: Camera connection lost!")
                    print(f"Attempting to reconnect to camera {original_camera_index}...")
                    # Try to reconnect using the SAME camera and settings
                    try:
                        if cap:
                            cap.release()
                        time.sleep(0.5)
                        # Reconnect using original camera index and backend (preserve your camera)
                        cap = cv2.VideoCapture(original_camera_index, original_backend)
                        if not cap.isOpened():
                            print("Failed to reconnect camera. Exiting...")
                            running = False
                            break
                        # Verify we can read frames
                        ret, test_frame = cap.read()
                        if not ret or test_frame is None:
                            print("Camera reconnected but cannot read frames. Exiting...")
                            running = False
                            break
                        print("Camera reconnected successfully!")
                        print(f"  Using camera's settings: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    except Exception as reconnect_error:
                        print(f"Reconnection error: {reconnect_error}")
                        running = False
                        break
                    continue
                
                # LATENCY REDUCTION: Drop old frames to catch up with camera (skip buffered frames)
                # Read and discard up to 2 old frames to reduce latency
                # Use grab() which is faster than read() as it doesn't decode
                for _ in range(2):
                    if not cap.grab():  # Grab frame without decoding (fast)
                        break  # No more frames to grab, break early
                
                # Read the latest frame
                ret, frame = cap.read()
                
                if not ret:
                    # Check if video file ended
                    if is_video_file:
                        print("Video file ended. Exiting...")
                        running = False
                        break
                    else:
                        print("Warning: Could not read frame from webcam, retrying...")
                        # Keep window open and retry - don't close on frame read error
                        time.sleep(0.1)  # Small delay before retrying
                        continue
                
                # Validate frame
                if frame is None or frame.size == 0:
                    print("Warning: Invalid frame received, skipping...")
                    continue
                
                # Ensure frame has valid dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    print("Warning: Frame has invalid dimensions, skipping...")
                    continue
                
                # Update human detection cache periodically to suppress human false positives
                if HUMAN_SUPPRESSION_ENABLED:
                    frames_since_human_detect += 1
                    if frames_since_human_detect >= HUMAN_DETECTION_INTERVAL:
                        human_boxes = detect_humans_in_frame(frame)
                        frames_since_human_detect = 0
                else:
                    human_boxes = []
                
                # Apply lightweight enhancement for LOW LATENCY
                # Only apply fast enhancement if enabled (Retinex is too slow)
                if USE_FAST_ENHANCEMENT:
                    frame = apply_fast_enhancement(frame)  # Lightweight CLAHE enhancement
                # Retinex is disabled by default (too slow, causes latency)
                # Can be enabled with 'r' key if clarity is more important than latency
                if RETINEX_ENABLED:
                    frame = apply_retinex(frame)  # Multi-Scale Retinex (SLOW - causes latency)
                
                # Flip frame horizontally for mirror effect (optional)
                # frame = cv2.flip(frame, 1)
                
                # Draw full-screen YOLO-style grid overlay (appears on entire frame)
                # This creates a dynamic grid overlay like YOLO training visualization
                # frame = draw_yolo_grid(frame, grid_size=args.grid_size, alpha=args.grid_alpha)  # Disabled - grid removed
                
                # Run YOLOv8 inference - optimized for accuracy
                # Use larger inference size for better detection accuracy
                # Use standard IoU threshold for NMS
                results = model.predict(
                    frame,
                    conf=confidence_threshold * 0.8,  # Use slightly lower threshold for model (we filter in draw_detections)
                    iou=0.45,                        # Standard IoU threshold for NMS
                    imgsz=INFERENCE_SIZE,             # Inference image size (640 for better accuracy)
                    verbose=False,                    # Disable verbose output
                    half=False,                       # Use full precision (disable half precision for compatibility)
                    agnostic_nms=False,               # Class-aware NMS for better pest detection
                    max_det=300                       # Allow more detections per frame
                )
                
                # Get current time for pulsing animation
                current_time = time.time()
                
                # Draw detections on frame (bounding boxes, labels, grid around each pest)
                # Grid appears around each detected pest with pulsing animation
                frame = draw_detections(
                    frame, 
                    results, 
                    confidence_threshold=confidence_threshold,
                    detection_timestamps=detection_timestamps,
                    current_time=current_time,
                    detected_pests=detected_pests,
                    human_boxes=human_boxes
                )
            except Exception as frame_error:
                # Handle frame processing errors - continue loop instead of exiting
                error_msg = str(frame_error)
                print(f"Warning: Error processing frame: {error_msg}")
                print("Continuing... (camera stays open)")
                # Check if camera is still valid
                if cap is not None and cap.isOpened():
                    # Try to read next frame
                    time.sleep(0.1)
                    continue
                else:
                    print("Camera connection lost during error handling!")
                    running = False
                    break
            
            # Calculate total detections
            total_detections = sum(len(r.boxes) for r in results)
            
            # Update detection status
            if total_detections > 0:
                detection_status = f"Detected: {total_detections} pest(s)"
            else:
                detection_status = "Detecting..."
            
            # Calculate FPS (smoother calculation)
            fps_counter += 1
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - fps_timer) / cv2.getTickFrequency()
            
            if elapsed_time >= 1.0:  # Update FPS every second
                fps_display = fps_counter / elapsed_time
                fps_counter = 0
                fps_timer = current_time
            
            # Display real-time FPS counter
            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Display detection status (e.g., "Detecting..." or "Detected: X pest(s)")
            status_color = (0, 255, 255) if total_detections > 0 else (128, 128, 128)
            cv2.putText(
                frame,
                detection_status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA
            )
            
            # Display detection count
            cv2.putText(
                frame,
                f"Pests: {total_detections}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Display confidence threshold
            cv2.putText(
                frame,
                f"Confidence: {confidence_threshold:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Display instructions on screen (camera stays ON until quit)
            cv2.putText(
                frame,
                "Press 'q' or ESC to quit - Camera stays ON",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Resize frame to match display window size (adapt to camera settings)
            height, width = frame.shape[:2]
            
            # Resize frame to match display window dimensions (adapt to camera resolution)
            if width != display_width or height != display_height:
                # Use INTER_AREA for downscaling (better quality) or INTER_LINEAR for upscaling
                if width > display_width or height > display_height:
                    display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                else:
                    display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame  # Use original frame if already matches display size
            
            # Save frame to video if enabled
            if video_writer is not None:
                video_writer.write(frame)  # Write original resolution frame with annotations
            
            # Validate display frame before showing
            if display_frame is None or display_frame.size == 0:
                print("Warning: Invalid display frame, skipping display...")
                continue
            
            # Display frame on screen using OpenCV imshow - this shows camera output
            # This line displays the camera feed and detection results on your screen
            # The window updates continuously to show real-time camera feed
            try:
                cv2.imshow(window_name, display_frame)
            except Exception as display_error:
                print(f"Warning: Error displaying frame: {display_error}")
                continue
            
            # Process keyboard input - window stays open until 'q' or ESC is explicitly pressed
            # waitKey(1) is non-blocking - keeps camera feed running and updating on screen
            # Must call waitKey for window to display new frames (camera output is visible here)
            # waitKey returns -1 on some systems when window is closed, 255 when no key pressed
            key_code = cv2.waitKey(1)
            if key_code == -1:
                # On some systems, -1 indicates window closed - verify before exiting
                try:
                    # Double-check by trying to get window property
                    window_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    if window_prop is not None and window_prop <= 0:
                        print("\nWindow was closed by user")
                        running = False
                        break
                except cv2.error:
                    # Window doesn't exist - it was closed
                    print("\nWindow was closed")
                    running = False
                    break
                except:
                    # Other error - window might still be open, continue
                    pass
            
            # Extract key code (mask to get ASCII value)
            key = key_code & 0xFF if key_code >= 0 else 255
            
            # Check if window was closed by clicking X button (check less frequently to avoid false positives)
            # Only check every 100 frames to reduce overhead and false detections on Windows
            # Note: On Windows, this check can sometimes return false positives, so we're very conservative
            if fps_counter % 100 == 0:
                try:
                    window_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    # window_prop is 1.0 when visible, 0.0 or negative when closed
                    # On Windows, sometimes this can return unexpected values, so be very lenient
                    # Only exit if we're absolutely sure the window is closed
                    if window_prop is not None and isinstance(window_prop, (int, float)) and window_prop <= 0:
                        # Double-check - wait a moment and check again to avoid false positives
                        time.sleep(0.05)
                        window_prop2 = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                        if window_prop2 is not None and isinstance(window_prop2, (int, float)) and window_prop2 <= 0:
                            print("\nWindow was closed by user (X button clicked)")
                            running = False
                            break
                except cv2.error:
                    # Window doesn't exist (closed), exit gracefully
                    print("\nWindow was closed")
                    running = False
                    break
                except:
                    # Ignore other errors and continue - window might still be open
                    # This prevents false positives on Windows
                    pass
            
            # Camera stays ON continuously - only quit when 'q', 'Q', or ESC is explicitly pressed
            # ESC key = 27, 'q' key = ord('q')
            # Note: waitKey returns 255 when no key is pressed
            if key != 255:  # A key was actually pressed
                if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q', or ESC key
                    print("\nQuit key ('q', 'Q', or ESC) pressed. Closing camera...")
                    running = False
                    break
                elif key == ord('s'):
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'captured_frame_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as: {filename}")
                elif key == ord('+') or key == ord('='):
                    confidence_threshold = min(0.95, confidence_threshold + 0.05)
                    print(f"Confidence threshold: {confidence_threshold:.2f}")
                elif key == ord('-'):
                    confidence_threshold = max(0.05, confidence_threshold - 0.05)
                    print(f"Confidence threshold: {confidence_threshold:.2f}")
                elif key == ord('r'):
                    # Toggle Retinex on/off (CLAHE remains enabled)
                    RETINEX_ENABLED = not RETINEX_ENABLED
                    if RETINEX_ENABLED:
                        status = "CLAHE + Retinex (maximum clarity)"
                        if RASPBERRY_PI_MODE:
                            print(f"Enhancement: {status} (WARNING: May reduce FPS on Pi!)")
                        else:
                            print(f"Enhancement: {status}")
                    else:
                        status = "CLAHE only (better FPS)"
                        print(f"Enhancement: {status}")
                elif key == ord('e'):
                    # Toggle all enhancement on/off completely
                    USE_FAST_ENHANCEMENT = not USE_FAST_ENHANCEMENT
                    if not USE_FAST_ENHANCEMENT:
                        RETINEX_ENABLED = False
                    status = "ON" if USE_FAST_ENHANCEMENT else "OFF"
                    enhancement_details = ""
                    if USE_FAST_ENHANCEMENT:
                        if RETINEX_ENABLED:
                            enhancement_details = " (CLAHE + Retinex)"
                        else:
                            enhancement_details = " (CLAHE only)"
                    print(f"Image enhancement: {status}{enhancement_details}")
            
            
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        print("Closing program...")
        running = False
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to continue... (window stays open)")
        # Continue running - don't exit on errors unless it's critical
        # The loop will continue and try to recover
    finally:
        # Only cleanup when we're actually exiting
        print("\nCleaning up...")
        if video_writer is not None:
            video_writer.release()
            print("Video saved successfully!")
        if cap is not None:
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("Window closed. Goodbye!")

if __name__ == '__main__':
    main()

