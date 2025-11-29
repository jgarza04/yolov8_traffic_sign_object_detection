import os
from pathlib import Path
import yaml

# === EDIT THIS: your dataset root ===
DATASET_ROOT = Path("/Users/juanpablogarza/Desktop/sign_detection.v11i.yolov8")  # <-- CHANGE THIS

# === ID mappings (old_id -> new_id) ===
OLD_TO_NEW = {
    3: 0,  # stop
    4: 1 ,  # yield
    1: 3,  # keepRight
    2: 2,
    0: 5,
}

NEW_CLASS_NAMES = ["stop", "yield", "doNotEnter", "keepRight", "speedLimit25", "finish"]

SPLITS = ["train", "valid", "test"]  # adjust if no test split


def remap_split(root: Path, split: str):
    lbl_dir = root / "labels" / split
    img_dir = root / "images" / split

    if not lbl_dir.exists():
        print(f"Skipping missing split: {split}")
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

            if old_id not in OLD_TO_NEW:
                continue

            new_id = OLD_TO_NEW[old_id]
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts) + "\n")

        if new_lines:
            with open(txt_path, "w") as f:
                f.writelines(new_lines)
            kept += 1
        else:
            # remove label file with no relevant objects
            txt_path.unlink()
            removed += 1

    print(f"  Kept: {kept}   Removed empty: {removed}")


def update_yaml(root: Path):
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        print("No data.yaml found, skipping update.")
        return

    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    y["train"] = str(root / "images" / "train")
    y["val"]   = str(root / "images" / "valid")
    if (root / "images" / "test").exists():
        y["test"] = str(root / "images" / "test")

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
