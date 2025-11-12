# Real-time Pest Detection System using YOLOv8

A complete system for detecting agricultural pests in real-time using YOLOv8 model with webcam feed.

## Dataset

- **Dataset Name**: Pest Detection (YOLOv8) – v1
- **Images**: 101 images (640×640 pixels)
- **Format**: YOLOv8 format
- **Classes**: 10 pest types
  1. Aphids
  2. Armyworm
  3. Beetle
  4. Bollworm
  5. Grasshopper
  6. Mosquito
  7. Stem borer
  8. Weevil
  9. Whitefly
  10. Fruit fly

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download your dataset from Roboflow**:
   
   Option A: Use the provided script
   ```bash
   python download_dataset.py
   ```
   
   Option B: Download manually from Roboflow and extract to `dataset/` folder
   
   Option C: Use Roboflow Python SDK
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
   dataset = project.version(1).download("yolov8")
   ```

4. **Verify dataset structure**:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/ (optional)
   └── labels/
       ├── train/
       ├── val/
       └── test/ (optional)
   ```

5. **Update `data.yaml`** with correct dataset path if needed

## Training

Train the YOLOv8 model on your pest detection dataset:

```bash
python train.py
```

The training will:
- Use YOLOv8n (nano) model by default
- Train for 100 epochs
- Save best model to `runs/train/pest_detection/weights/best.pt`
- Generate training plots and metrics

### Training Parameters

You can modify training parameters in `train.py`:
- `epochs`: Number of training epochs (default: 100)
- `imgsz`: Image size (default: 640)
- `batch`: Batch size (default: 16)
- `device`: 'cuda' for GPU, 'cpu' for CPU

## Real-time Detection

Run the real-time pest detection on webcam:

```bash
python detect_realtime.py
```

### Features

- Real-time pest detection from webcam feed
- Bounding boxes around detected pests
- Class name and confidence score displayed above each detection
- Color-coded bounding boxes for different pest classes
- FPS counter
- Detection count display

### Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

### Model Selection

The script will automatically use:
1. Trained model: `runs/train/pest_detection/weights/best.pt` (if available)
2. Fallback: Pretrained YOLOv8n model

## Project Structure

```
pest-detection-system/
├── data.yaml              # Dataset configuration
├── train.py               # Training script
├── detect_realtime.py     # Real-time detection script
├── download_dataset.py     # Roboflow dataset downloader
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── dataset/              # Dataset folder (create after download)
│   ├── images/
│   └── labels/
└── runs/                  # Training results (created after training)
    └── train/
        └── pest_detection/
            └── weights/
                └── best.pt
```

## Performance Tips

1. **For faster inference**: Use YOLOv8n (nano) or YOLOv8s (small)
2. **For better accuracy**: Use YOLOv8m (medium), YOLOv8l (large), or YOLOv8x (xlarge)
3. **Adjust confidence threshold** in `detect_realtime.py` (default: 0.25)
4. **Use GPU** for faster training and inference (set `device='cuda'`)

## Troubleshooting

### Webcam not working
- Check if webcam is connected and not used by another application
- Try changing camera index in `detect_realtime.py`: `cv2.VideoCapture(1)`

### Dataset not found
- Ensure dataset is in `dataset/` folder
- Check `data.yaml` has correct paths
- Verify Roboflow dataset structure matches expected format

### Model not found
- Train the model first using `python train.py`
- Or ensure pretrained model `yolov8n.pt` is available

### CUDA/GPU issues
- Install CUDA-compatible PyTorch version
- Or use CPU: change `device='cpu'` in `train.py`

## License

This project is for agricultural pest detection purposes.

## Contributing

Feel free to improve this system by:
- Adding more pest classes
- Optimizing detection speed
- Improving accuracy
- Adding image/video file detection support

