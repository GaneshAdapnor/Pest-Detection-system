import os
import sys
from roboflow import Roboflow


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    data_root_env = os.environ.get("DATASETS_DIR")
    if data_root_env:
        data_root = os.path.abspath(os.path.expanduser(data_root_env))
    else:
        data_root = os.path.join(os.path.expanduser("~"), "pest_datasets")
    os.makedirs(data_root, exist_ok=True)

    print(f"Using datasets directory: {data_root}")

    rf = Roboflow(api_key=api_key)

    print("Downloading Pest Detection YOLO dataset...")
    project1 = rf.workspace("systems-india").project("pest-detection-yolo-rfmcm")
    dataset1 = project1.version(1).download("yolov8", location=os.path.join(data_root, "pest-detection-yolo-rfmcm"))
    print(f"Saved to: {dataset1.location}")

    print("Downloading YOLOv5 Insect Detection dataset...")
    project2 = rf.workspace("school-oohsw").project("yolov5-insect-detection")
    dataset2 = project2.version(3).download("yolov8", location=os.path.join(data_root, "yolov5-insect-detection"))
    print(f"Saved to: {dataset2.location}")


if __name__ == "__main__":
    main()

