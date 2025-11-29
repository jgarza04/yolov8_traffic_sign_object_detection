import os
from pathlib import Path

# === EDIT THIS: your dataset root ===
# This should point to the folder containing data.yaml, train, valid, and test
DATASET_ROOT = Path("/Users/juanpablogarza/Desktop/sign_detection.v9i.yolov8")

# Adjust splits if necessary
SPLITS = ["train", "valid", "test"]

def cleanup_split(root: Path, split: str):
    """
    Identifies and deletes image files in a split that do not have 
    a corresponding label file in the YOLO format.
    """
    # 1. Define paths based on your confirmed structure
    lbl_dir = root / split / "labels"
    img_dir = root / split / "images"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"⚠️ Skipping split '{split}': Missing images or labels directory.")
        return

    # 2. Get the list of ALL images and ALL labels
    all_images = set([f.stem for f in img_dir.glob("*") if f.is_file()])
    all_labels = set([f.stem for f in lbl_dir.glob("*.txt")])

    print(f"\n[{split}] Found {len(all_images)} images and {len(all_labels)} label files.")

    # 3. Determine which images are UNLABELED
    # Images whose stem is NOT in the set of labels are candidates for deletion.
    unlabeled_images = all_images - all_labels
    
    deleted_count = 0

    if unlabeled_images:
        print(f"🗑️ Deleting {len(unlabeled_images)} unlabeled image files...")
        
        for img_stem in unlabeled_images:
            # The script attempts to find the image file, regardless of extension (.jpg, .png, etc.)
            
            # Find the actual file path using a glob that matches any extension
            # Note: glob finds all files matching the stem followed by anything
            img_path_candidates = list(img_dir.glob(f"{img_stem}.*"))

            for img_path in img_path_candidates:
                if img_path.is_file():
                    try:
                        img_path.unlink()  # Permanently deletes the file
                        deleted_count += 1
                        # print(f"  Deleted: {img_path.name}") # Uncomment for debugging
                    except Exception as e:
                        print(f"Error deleting {img_path.name}: {e}")
                        
    else:
        print("✅ All image files have corresponding label files. No action needed.")

    if deleted_count > 0:
        print(f"--- Summary for {split} ---")
        print(f"Total deleted: {deleted_count}")
        print(f"Remaining images: {len(all_images) - deleted_count}")
        
    if len(all_labels) != (len(all_images) - deleted_count):
        print("⚠️ WARNING: A discrepancy exists. Some labels might be missing their image.")


def main():
    root = DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"🔍 Starting cleanup process in: {root}")

    for split in SPLITS:
        cleanup_split(root, split)

    print("\n✅ DONE — Dataset cleanup complete.")


if __name__ == "__main__":
    main()