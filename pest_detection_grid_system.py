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

CONFIDENCE_THRESHOLD: float = 0.65
NMS_IOU_THRESHOLD: float = 0.45
FULL_GRID_SIZE: int = 40
MINI_GRID_CELLS: int = 4


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
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# Detection Helpers
# ---------------------------------------------------------------------------

def extract_detections(results, confidence_threshold: float) -> List[Detection]:
    """Convert YOLO results into filtered Detection objects."""
    detections: List[Detection] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf.item())
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls.item())
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue  # Ignore classes outside our pest list

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
                               grid_cells: int = MINI_GRID_CELLS) -> np.ndarray:
    """Draw bounding boxes, labels, and mini-grids for detections."""
    for det in detections:
        class_name = CLASS_NAMES[det.class_id]
        color = CLASS_COLORS[class_name]
        x1, y1, x2, y2 = det.bbox

        # Mini grid inside bounding box
        draw_box_grid(frame, x1, y1, x2, y2, cells=grid_cells, color=color, thickness=1)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        # Text label with background
        label = f"{class_name} {det.confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x1
        text_y = max(20, y1 - 10)

        bg_x1 = text_x
        bg_y1 = text_y - text_height - baseline
        bg_x2 = text_x + text_width + 6
        bg_y2 = text_y + baseline

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
            (text_x + 3, text_y - 3),
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
        help="Confidence threshold for displaying detections (default: 0.65).",
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

    window_title = "Pest Detection Grid System"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from source. Exiting loop.")
                break

            frame_height, frame_width = frame.shape[:2]

            # Run inference on current frame
            results = model.predict(
                frame,
                conf=args.confidence,
                iou=NMS_IOU_THRESHOLD,
                imgsz=640,
                verbose=False,
                half=False,
                agnostic_nms=False,
                classes=list(range(len(CLASS_NAMES))),  # Restrict to pest classes
            )

            detections = extract_detections(results, args.confidence)

            if detections:
                # CLAHE enhancement first
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    apply_clahe_to_roi(frame, x1, y1, x2, y2)

                # Draw overlays and annotations
                draw_full_grid(frame, grid_size=args.grid_size)
                draw_detection_annotations(frame, detections)

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

