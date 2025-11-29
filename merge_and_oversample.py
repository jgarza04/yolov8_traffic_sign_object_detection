import os
import shutil
import argparse

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def copy_split(new_root, base_root, split, oversample=1, prefix="f1_"):
    """
    Copy images+labels from:
        new_root/{split}/images, new_root/{split}/labels
    into:
        base_root/images/{split}, base_root/labels/{split}

    For train split, you can oversample by copying each example `oversample` times.
    """
    src_img_dir = os.path.join(new_root, split, "images")
    src_lbl_dir = os.path.join(new_root, split, "labels")

    dst_img_dir = os.path.join(base_root, "images", split)
    dst_lbl_dir = os.path.join(base_root, "labels", split)

    if not os.path.isdir(src_img_dir) or not os.path.isdir(src_lbl_dir):
        print(f"[WARN] Skipping split '{split}' — missing images or labels folder.")
        return

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    img_files = [
        f for f in os.listdir(src_img_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    img_files.sort()

    n_copied = 0

    for img_name in img_files:
        stem, ext = os.path.splitext(img_name)
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, stem + ".txt")

        if not os.path.isfile(src_lbl_path):
            print(f"[WARN] No label for {img_name}, skipping.")
            continue

        repeat = oversample if split == "train" else 1

        for k in range(repeat):
            # f1_<original>_dup1, f1_<original>_dup2, ...
            suffix = "" if k == 0 else f"_dup{k}"
            new_stem = f"{prefix}{stem}{suffix}"

            dst_img_path = os.path.join(dst_img_dir, new_stem + ext)
            dst_lbl_path = os.path.join(dst_lbl_dir, new_stem + ".txt")

            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(src_lbl_path, dst_lbl_path)
            n_copied += 1

    print(f"[OK] Split '{split}': copied {n_copied} image+label pairs "
          f"({'x'+str(oversample) if split=='train' else 'no oversample'})")


def main():
    parser = argparse.ArgumentParser(
        description="Merge new F1TENTH-like YOLOv8 dataset into base dataset and oversample train."
    )
    parser.add_argument("--new-root", required=True,
                        help="Path to NEW dataset root (has train/valid/test with images+labels).")
    parser.add_argument("--base-root", required=True,
                        help="Path to BASE dataset root (traffic_sign_dataset).")
    parser.add_argument("--oversample", type=int, default=2,
                        help="Oversampling factor for TRAIN split (default: 2).")
    parser.add_argument("--prefix", type=str, default="f1_",
                        help="Filename prefix for new images/labels to avoid collisions.")

    args = parser.parse_args()

    print(f"New dataset root : {args.new_root}")
    print(f"Base dataset root: {args.base_root}")
    print(f"Oversample factor (train): {args.oversample}")
    print()

    for split in ["train", "valid", "test"]:
        copy_split(args.new_root, args.base_root, split,
                   oversample=args.oversample, prefix=args.prefix)

    print("\n[NOTE] Now delete YOLO cache files so it sees the new data:")
    print(f"  rm {os.path.join(args.base_root, 'labels', 'train.cache')}  (if exists)")
    print(f"  rm {os.path.join(args.base_root, 'labels', 'valid.cache')}  (if exists)")
    print("\nThen fine-tune starting from your best model.")


if __name__ == "__main__":
    main()