"""
YOLOv8 Training Script for Pest Detection
Trains a YOLOv8 model on the Roboflow "Pest Detection (YOLOv8) – v1" dataset
"""

from ultralytics import YOLO
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings during training

def train_pest_detection_model():
    """
    Train YOLOv8 model on pest detection dataset
    """
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("ERROR: data.yaml not found!")
        print("Please create data.yaml with your dataset configuration.")
        return
    
    # Initialize YOLOv8 model
    # Using YOLOv8n (nano) for faster training and inference
    # Options: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
    print("Initializing YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Start from pretrained weights
    
    # Detect available device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU for training (GPU not available)")
    
    # Adjust batch size based on device
    batch_size = 16 if device == 'cuda' else 8  # Smaller batch for CPU
    
    # Train the model
    print("Starting training...")
    print("\n" + "=" * 70)
    print("Training will display progress here...")
    print("=" * 70 + "\n")
    
    try:
        results = model.train(
            data='data.yaml',          # Dataset configuration file
            epochs=100,                 # Number of training epochs
            imgsz=640,                  # Image size (must match dataset)
            batch=batch_size,           # Batch size (adjusted for CPU/GPU)
            name='pest_detection',      # Project name
            device=device,              # Use GPU if available, else 'cpu'
            patience=50,                # Early stopping patience
            save=True,                 # Save checkpoints
            plots=True,                # Generate training plots
            verbose=True,               # Verbose output
            project='runs/train',       # Project directory
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nTrying to disable TensorBoard to avoid compatibility issues...")
        # Try without plots to avoid TensorBoard issues
        results = model.train(
            data='data.yaml',
            epochs=100,
            imgsz=640,
            batch=batch_size,
            name='pest_detection',
            device=device,
            patience=50,
            save=True,
            plots=False,  # Disable plots to avoid TensorBoard
            verbose=True,
            project='runs/train',
        )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Results saved at: {results.save_dir}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model

if __name__ == '__main__':
    print("=" * 60)
    print("Pest Detection Model Training")
    print("=" * 60)
    
    # Check dataset directory exists
    dataset_path = './Pest detection -YOLOv8-.v1i.yolov8'
    if not os.path.exists(dataset_path):
        print(f"\nWARNING: Dataset directory not found: {dataset_path}")
        print("\nPlease ensure your dataset is available.")
        print("The dataset should be in 'Pest detection -YOLOv8-.v1i.yolov8' directory")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    train_pest_detection_model()