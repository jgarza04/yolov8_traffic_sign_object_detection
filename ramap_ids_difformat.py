import os
from pathlib import Path
import yaml

# === EDIT THIS: your dataset root ===
# This should point to the folder containing data.yaml, train, valid, and test
DATASET_ROOT = Path("/Users/juanpablogarza/Desktop/sign_detection.v11i.yolov8")  # <-- KEEP THIS

# === ID mappings (old_id -> new_id) ===
OLD_TO_NEW = {
    3: 0,  # stop
    4: 1 ,  # yield
    1: 3,  # keepRight
    2: 2,  # doNotEnter
    0: 5,  # speedLimit25 (assuming based on your new names)
}

NEW_CLASS_NAMES = ["stop", "yield", "doNotEnter", "keepRight", "speedLimit25", "finish"]

SPLITS = ["train", "valid", "test"]


def remap_split(root: Path, split: str):
    """
    MODIFIED: Looks for the 'labels' directory *inside* the split folder.
    Expected path: DATASET_ROOT / train / labels / *.txt
    """
    # Construct the path to the split folder (e.g., DATASET_ROOT/train)
    split_dir = root / split
    
    # Construct the path to the labels folder (e.g., DATASET_ROOT/train/labels)
    lbl_dir = split_dir / "labels"
    # Construct the path to the images folder (e.g., DATASET_ROOT/train/images)
    img_dir = split_dir / "images" 

    if not lbl_dir.exists():
        print(f"Skipping missing split: {split} (labels folder not found at {lbl_dir})")
        return

    txt_files = list(lbl_dir.glob("*.txt"))
    print(f"\n[{split}] remapping {len(txt_files)} label files...")

    kept, removed = 0, 0

    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                continue

            # Skip the bounding box if the old class ID is not in our mapping
            if old_id not in OLD_TO_NEW:
                continue

            new_id = OLD_TO_NEW[old_id]
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts) + "\n")

        # Overwrite the file with the remapped lines
        if new_lines:
            with open(txt_path, "w") as f:
                f.writelines(new_lines)
            kept += 1
        else:
            # Remove label file if it contained objects that were all removed
            txt_path.unlink()
            removed += 1

    print(f"  Kept: {kept}   Removed empty: {removed}")


def update_yaml(root: Path):
    """
    MODIFIED: Updates the paths in data.yaml to point to the images folders 
    inside the split directories (train/images, valid/images, etc.).
    """
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        print("No data.yaml found, skipping update.")
        return

    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    # Update paths to point to the correct structure: DATASET_ROOT/train/images
    y["train"] = str(root / "train" / "images")
    y["val"]   = str(root / "valid" / "images")
    if (root / "test" / "images").exists():
        y["test"] = str(root / "test" / "images")

    # Update class count and names
    y["nc"] = len(NEW_CLASS_NAMES)
    y["names"] = NEW_CLASS_NAMES

    with open(yaml_path, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)

    print(f"\nUpdated YAML at {yaml_path}")


def main():
    root = DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"📁 Remapping dataset in: {root}")

    for split in SPLITS:
        remap_split(root, split)

    update_yaml(root)

    print("\n✅ DONE — All class IDs have been remapped.")


if __name__ == "__main__":
    main()