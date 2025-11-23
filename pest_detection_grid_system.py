"""
Real-time Pest Detection System with YOLOv8, OpenCV, and CLAHE-enhanced overlays.

Features:
    - Full-screen YOLO-style grid overlay
    - CLAHE enhancement inside each detected pest bounding box
    - 4×4 mini-grid drawn within every pest bounding box
    - Colored bounding boxes per pest class with labels and confidences
    - FPS counter and smooth real-time display

Requirements:
    pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PestColor = Tuple[int, int, int]

CLASS_NAMES: List[str] = [
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
]

CLASS_COLORS: Dict[str, PestColor] = {
    "Ants": (0x00, 0xFF, 0x00),          # Green
    "Bees": (0x00, 0xFF, 0xFF),          # Yellow
    "Beetles": (0x00, 0x00, 0xFF),       # Red
    "Caterpillars": (0x00, 0xA5, 0xFF),  # Orange
    "Earthworms": (0x80, 0x00, 0x80),    # Purple
    "Earwigs": (0xC0, 0xC0, 0xC0),       # Silver
    "Grasshoppers": (0x00, 0xD7, 0xFF),  # Yellow (Gold)
    "Moths": (0xFF, 0x00, 0xFF),         # Magenta
    "Slugs": (0x00, 0x64, 0x00),         # Dark Green
    "Snails": (0xCB, 0xC0, 0xFF),        # Pink
    "Wasps": (0xFF, 0x00, 0x00),         # Blue
    "Weevils": (0xFF, 0xFF, 0xFF),       # White
}

# Ensure colors are in BGR order for OpenCV drawing
CLASS_COLORS = {
    name: (color[0], color[1], color[2]) for name, color in CLASS_COLORS.items()
}

CONFIDENCE_THRESHOLD: float = 0.70  # Higher threshold for more accurate detections
NMS_IOU_THRESHOLD: float = 0.50  # Slightly higher to allow closer detections
FULL_GRID_SIZE: int = 40
MINI_GRID_CELLS: int = 4
INFERENCE_SIZE: int = 640  # Optimized for speed while maintaining accuracy


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Drawing Utilities
# ---------------------------------------------------------------------------

def draw_full_grid(frame: np.ndarray, grid_size: int = FULL_GRID_SIZE,
                   color: Tuple[int, int, int] = (190, 190, 190),
                   alpha: float = 0.25) -> np.ndarray:
    """Overlay a YOLO-style grid across the entire frame."""
    if grid_size <= 0:
        return frame

    overlay = frame.copy()
    height, width = frame.shape[:2]

    for x in range(0, width, grid_size):
        cv2.line(overlay, (x, 0), (x, height), color, 1, lineType=cv2.LINE_AA)
    for y in range(0, height, grid_size):
        cv2.line(overlay, (0, y), (width, y), color, 1, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_box_grid(frame: np.ndarray,
                  x1: int,
                  y1: int,
                  x2: int,
                  y2: int,
                  cells: int = MINI_GRID_CELLS,
                  color: Tuple[int, int, int] = (200, 200, 200),
                  thickness: int = 1) -> np.ndarray:
    """Draw a mini-grid inside the bounding box extent."""
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0 or cells <= 0:
        return frame

    step_x = width / cells
    step_y = height / cells

    for i in range(1, cells):
        px = int(round(x1 + step_x * i))
        cv2.line(frame, (px, y1), (px, y2), color, thickness, lineType=cv2.LINE_AA)

    for j in range(1, cells):
        py = int(round(y1 + step_y * j))
        cv2.line(frame, (x1, py), (x2, py), color, thickness, lineType=cv2.LINE_AA)

    return frame


def apply_clahe_to_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       clip_limit: float = 2.0,
                       tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE enhancement only inside the specified bounding box."""
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    frame[y1:y2, x1:x2] = enhanced_bgr
    return frame


# ---------------------------------------------------------------------------
# Model Utilities
# ---------------------------------------------------------------------------

def discover_best_model(specified_path: Optional[str]) -> str:
    """Resolve the most appropriate YOLO model weights to load."""
    if specified_path:
        model_path = Path(specified_path).expanduser().resolve()
        if model_path.exists():
            return str(model_path)
        print(f"[WARN] Specified model not found at {model_path}. Falling back.", file=sys.stderr)

    # Candidate search priorities
    candidates: List[Path] = []

    # 1. Local best.pt in current directory
    local_best = Path("best.pt")
    if local_best.exists():
        candidates.append(local_best.resolve())

    # 2. Latest run in runs/train/**/weights/best.pt
    runs_root = Path("runs/train")
    if runs_root.exists():
        for path in runs_root.glob("**/weights/best.pt"):
            candidates.append(path.resolve())

    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)

    # 3. Fallback to canonical YOLOv8n weights
    return "yolov8n.pt"


def load_model(specified_model: Optional[str]) -> YOLO:
    """Load YOLO model with preference for trained weights."""
    model_path = discover_best_model(specified_model)
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Verify model class names match our expected classes
    if hasattr(model, 'names') and model.names:
        model_class_names = [model.names[i] for i in sorted(model.names.keys())]
        print(f"[INFO] Model has {len(model_class_names)} classes: {model_class_names}")
        
        # Check if they match our CLASS_NAMES
        if model_class_names != CLASS_NAMES:
            print(f"[WARNING] Model class names don't match hardcoded CLASS_NAMES!")
            print(f"  Model: {model_class_names}")
            print(f"  Code:  {CLASS_NAMES}")
            print(f"[INFO] Using model's class names for detection.")
    
    return model


# ---------------------------------------------------------------------------
# Detection Helpers
# ---------------------------------------------------------------------------

def extract_detections(results, confidence_threshold: float, model_names: Optional[Dict[int, str]] = None) -> List[Detection]:
    """Convert YOLO results into filtered Detection objects."""
    detections: List[Detection] = []
    
    # Use model's class names if available, otherwise use hardcoded CLASS_NAMES
    if model_names:
        # Convert model.names dict to list
        max_class_id = max(model_names.keys()) if model_names else len(CLASS_NAMES) - 1
        class_names_list = [model_names.get(i, f"Class_{i}") for i in range(max_class_id + 1)]
    else:
        class_names_list = CLASS_NAMES

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf.item())
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls.item())
            if cls_id < 0 or cls_id >= len(class_names_list):
                continue  # Ignore classes outside our pest list
            
            # Validate class name is in our pest list
            detected_class_name = class_names_list[cls_id]
            if detected_class_name not in CLASS_NAMES:
                continue  # Skip if not a pest class

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

    return detections


def draw_detection_annotations(frame: np.ndarray,
                               detections: Sequence[Detection],
                               grid_cells: int = MINI_GRID_CELLS,
                               model_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """Draw bounding boxes, labels, and mini-grids for detections."""
    # Use model's class names if available
    if model_names:
        max_class_id = max(model_names.keys()) if model_names else len(CLASS_NAMES) - 1
        class_names_list = [model_names.get(i, f"Class_{i}") for i in range(max_class_id + 1)]
    else:
        class_names_list = CLASS_NAMES
    
    for det in detections:
        # Get class name from the appropriate source
        if det.class_id < len(class_names_list):
            class_name = class_names_list[det.class_id]
        else:
            class_name = CLASS_NAMES[det.class_id] if det.class_id < len(CLASS_NAMES) else f"Class_{det.class_id}"
        
        # Ensure class_name exists in CLASS_COLORS, fallback to first color if not
        if class_name not in CLASS_COLORS:
            print(f"[WARNING] Class '{class_name}' not found in CLASS_COLORS, using default color")
            color = (255, 255, 255)  # White as fallback
        else:
            color = CLASS_COLORS[class_name]
        x1, y1, x2, y2 = det.bbox

        # Expand grid area by 10% for better visibility
        width = x2 - x1
        height = y2 - y1
        expand_factor = 0.10
        pad_x = int(width * expand_factor / 2)
        pad_y = int(height * expand_factor / 2)
        
        grid_x1 = max(0, x1 - pad_x)
        grid_y1 = max(0, y1 - pad_y)
        grid_x2 = min(frame.shape[1], x2 + pad_x)
        grid_y2 = min(frame.shape[0], y2 + pad_y)

        # Draw prominent grid around the pest (expanded area)
        draw_box_grid(frame, grid_x1, grid_y1, grid_x2, grid_y2, 
                     cells=grid_cells, color=color, thickness=2)
        
        # Draw border around the grid area
        cv2.rectangle(frame, (grid_x1, grid_y1), (grid_x2, grid_y2), color, 3, lineType=cv2.LINE_AA)

        # Mini grid inside bounding box (original area)
        draw_box_grid(frame, x1, y1, x2, y2, cells=grid_cells, color=color, thickness=1)

        # Bounding box (thicker for visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

        # Text label with background - make it more prominent
        label = f"{class_name} {det.confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.6 for better visibility
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x1
        text_y = max(30, y1 - 10)

        bg_x1 = text_x - 4
        bg_y1 = text_y - text_height - baseline - 4
        bg_x2 = text_x + text_width + 8
        bg_y2 = text_y + baseline + 4

        # Draw semi-transparent background for better readability
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw pest name text
        cv2.putText(
            frame,
            label,
            (text_x + 4, text_y - 4),
            font,
            font_scale,
            color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Pest Detection System with YOLOv8.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a YOLOv8 model (default: auto-detect best.pt or fall back to yolov8n.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video file path (default: 0).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=FULL_GRID_SIZE,
        help="Grid size for full-screen overlay (default: 40).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for displaying detections (default: {CONFIDENCE_THRESHOLD}).",
    )
    return parser.parse_args()


def resolve_video_source(source_arg: str) -> int | str:
    """Interpret the source argument as camera index or file path."""
    if len(source_arg) == 1 and source_arg.isdigit():
        return int(source_arg)
    if source_arg.isdigit():
        # multi-digit but purely numeric -> treat as camera index
        return int(source_arg)
    return source_arg


def main() -> None:
    args = parse_args()
    model = load_model(args.model)

    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"[INFO] Using device: {device.upper()}")

    source = resolve_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {source}")
        sys.exit(1)
    
    # Optimize camera for low latency (if camera source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lowest latency
        # Don't modify resolution - use camera defaults as requested

    window_title = "Pest Detection Grid System"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    last_time = time.time()
    fps = 0.0

    try:
        while True:
            # Clear frame buffer to reduce latency - grab latest frame
            for _ in range(2):  # Skip up to 2 buffered frames
                cap.grab()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from source. Exiting loop.")
                break

            frame_height, frame_width = frame.shape[:2]

            # Run inference with optimized settings for SPEED
            # Fast detection when pests are spotted
            results = model.predict(
                frame,
                conf=args.confidence * 0.5,  # Lower threshold for model, filter higher in post-processing
                iou=NMS_IOU_THRESHOLD,
                imgsz=INFERENCE_SIZE,  # 640px for fast inference
                verbose=False,
                half=False,  # Keep full precision for accuracy
                agnostic_nms=False,
                max_det=100,  # Reduced from 300 for faster NMS processing
                augment=False,  # Disabled for speed
                stream=False,  # Single frame mode for lowest latency
            )

            # Get model's class names for proper mapping
            model_names = model.names if hasattr(model, 'names') and model.names else None
            
            detections = extract_detections(results, args.confidence, model_names=model_names)

            if detections:
                # CLAHE enhancement first
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    apply_clahe_to_roi(frame, x1, y1, x2, y2)

                # Draw overlays and annotations
                draw_full_grid(frame, grid_size=args.grid_size)
                draw_detection_annotations(frame, detections, model_names=model_names)

            # FPS measurement
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed > 0:
                fps = (fps * 0.9) + (0.1 * (1.0 / elapsed))
            last_time = current_time

            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

