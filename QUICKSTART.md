# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download Your Dataset

### Option 1: Using Roboflow (Recommended)
```bash
pip install roboflow
python download_dataset.py
```

### Option 2: Manual Download
- Go to your Roboflow project
- Export dataset in YOLOv8 format
- Extract to `dataset/` folder

## 3. Verify Dataset Structure

```bash
python check_dataset.py
```

This will verify:
- Correct folder structure
- Image and label file counts
- Label file format

## 4. Train the Model

```bash
python train.py
```

Training will:
- Save best model to `runs/train/pest_detection/weights/best.pt`
- Generate training plots and metrics
- Take 30-60 minutes on GPU (longer on CPU)

## 5. Run Real-time Detection

```bash
python detect_realtime.py
```

Press:
- **'q'** to quit
- **'s'** to save current frame

## 6. Test on Images/Videos (Optional)

```bash
# Detect in an image
python detect_batch.py path/to/image.jpg

# Detect in a video
python detect_batch.py path/to/video.mp4
```

## Troubleshooting

### "Dataset not found"
- Make sure dataset is in `dataset/` folder
- Check `data.yaml` has correct paths
- Run `python check_dataset.py` to verify structure

### "Model not found"
- Train the model first: `python train.py`
- Or the script will use pretrained YOLOv8n model

### "Webcam not working"
- Check webcam is connected
- Try changing camera index in `detect_realtime.py`: `cv2.VideoCapture(1)`

## Next Steps

- Adjust confidence threshold in `detect_realtime.py` (default: 0.25)
- Train with different YOLOv8 models (nano/small/medium/large)
- Add more training data to improve accuracy

