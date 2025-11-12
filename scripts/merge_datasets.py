import os
import shutil
import yaml
from pathlib import Path


OUTPUT_ROOT = Path("datasets") / "merged_pests"


def load_dataset_info(base_path):
    data_yaml_path = base_path / "data.yaml"
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]
    # normalise to list form if dict provided
    if isinstance(names, dict):
        # assume keys are ints
        names = [names[k] for k in sorted(names)]
    return {
        "base": base_path,
        "names": names,
        "splits": {
            "train": base_path / "train",
            "val": base_path / ("valid" if (base_path / "valid").exists() else "val"),
            "test": base_path / "test" if (base_path / "test").exists() else None,
        },
    }


def build_class_index(datasets):
    combined_names = []
    name_to_index = {}
    for dataset in datasets:
        for name in dataset["names"]:
            if name not in name_to_index:
                name_to_index[name] = len(combined_names)
                combined_names.append(name)
    return combined_names, name_to_index


def rewrite_label_file(src_label, dst_label, index_map):
    with open(src_label, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    rewritten = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        orig_idx = int(parts[0])
        if orig_idx not in index_map:
            raise ValueError(f"No mapped class index found for {src_label}, class {orig_idx}")
        new_idx = index_map[orig_idx]
        parts[0] = str(new_idx)
        rewritten.append(" ".join(parts))
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(rewritten) + ("\n" if rewritten else ""))


def copy_split(dataset, split_name, output_root, name_to_index, combined_names):
    split_path = dataset["splits"][split_name]
    if split_path is None or not split_path.exists():
        return
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images or labels directory for split '{split_name}' in {dataset['base']}")

    all_files = list(images_dir.glob("*"))
    index_map = {i: name_to_index[name] for i, name in enumerate(dataset["names"])}

    target_images = output_root / split_name / "images"
    target_labels = output_root / split_name / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    for image_path in all_files:
        if not image_path.is_file():
            continue
        rel_name = image_path.name
        dst_image_path = target_images / rel_name

        counter = 1
        while dst_image_path.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            dst_image_path = target_images / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.copy2(image_path, dst_image_path)

        label_name = image_path.with_suffix(".txt").name
        src_label_path = labels_dir / label_name
        if not src_label_path.exists():
            raise FileNotFoundError(f"Label file missing: {src_label_path}")
        dst_label_name = dst_image_path.with_suffix(".txt").name
        dst_label_path = target_labels / dst_label_name
        rewrite_label_file(src_label_path, dst_label_path, index_map)


def main():
    project_dataset = load_dataset_info(Path("Pest detection -YOLOv8-.v1i.yolov8"))
    roboflow_dataset1 = load_dataset_info(Path("C:/pest_datasets/pest-detection-yolo-rfmcm"))
    roboflow_dataset2 = load_dataset_info(Path("C:/pest_datasets/yolov5-insect-detection"))

    datasets = [project_dataset, roboflow_dataset1, roboflow_dataset2]

    combined_names, name_to_index = build_class_index(datasets)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for split in ["train", "val", "test"]:
            copy_split(dataset, split, OUTPUT_ROOT, name_to_index, combined_names)

    merged_yaml = {
        "path": str(OUTPUT_ROOT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(combined_names),
        "names": {i: name for i, name in enumerate(combined_names)},
    }

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    merged_yaml_path = config_dir / "merged_pests.yaml"
    with open(merged_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_yaml, f, sort_keys=False, allow_unicode=False)

    print(f"Merged dataset created at: {OUTPUT_ROOT}")
    print(f"Config written to: {merged_yaml_path}")


if __name__ == "__main__":
    main()

