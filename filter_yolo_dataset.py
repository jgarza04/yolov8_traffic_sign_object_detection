import argparse, os, shutil, yaml
from pathlib import Path
from tqdm import tqdm

def load_names(data_yaml):
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)
    # names can be a list or dict {id:name}
    names = y.get("names")
    if isinstance(names, dict):
        # convert id->name dict to list ordered by id
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names, y

def ensure_dirs(root_out, splits):
    for sp in splits:
        (root_out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root_out / "labels" / sp).mkdir(parents=True, exist_ok=True)

def filter_split(root_in, root_out, split, keep_ids):
    img_dir = root_in / split / "images"
    lbl_dir = root_in / split / "labels"
    kept, skipped = 0, 0

    # Accept common image suffixes
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]

    for img_path in tqdm(imgs, desc=f"[{split}] filtering"):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            skipped += 1
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        keep_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: 
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid in keep_ids:
                keep_lines.append(line)

        if keep_lines:
            # copy image + write filtered labels
            out_img = root_out / "images" / split / img_path.name
            out_lbl = root_out / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, out_img)
            with open(out_lbl, "w") as f:
                f.writelines(keep_lines)
            kept += 1
        else:
            skipped += 1

    return kept, skipped

def write_data_yaml(out_root, base_yaml_dict, splits, new_names):
    y = {}
    # point paths to new structure (relative)
    y["train"] = str(out_root / "images" / "train")
    # keep user's split naming (valid vs val). If both exist, prefer valid.
    if "valid" in splits:
        y["val"] = str(out_root / "images" / "valid")
    else:
        y["val"] = str(out_root / "images" / "val")  # just in case
    if "test" in splits:
        y["test"] = str(out_root / "images" / "test")
    y["nc"] = len(new_names)
    y["names"] = new_names
    # preserve optional metadata if you want (commented to keep it clean)
    # for k in ("roboflow", "license"):
    #     if k in base_yaml_dict: y[k] = base_yaml_dict[k]

    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser(description="Filter YOLOv8 dataset to two classes")
    ap.add_argument("--root", required=True,
                    help="Path to dataset root that contains train/ valid/ test/ and data.yaml")
    ap.add_argument("--keep", nargs="+", default=["SS", "yield sign"],
                    help='Class names to keep (default: "SS" "yield sign")')
    args = ap.parse_args()

    root_in = Path(args.root)
    data_yaml = root_in / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}")

    names, base_yaml = load_names(data_yaml)
    if not names:
        raise RuntimeError("Could not read class names from data.yaml")

    # Map keep class names -> ids
    name_to_id = {n: i for i, n in enumerate(names)}
    missing = [k for k in args.keep if k not in name_to_id]
    if missing:
        raise ValueError(f"These class names are not in data.yaml names: {missing}\nFound names: {names}")

    keep_ids = {name_to_id[k] for k in args.keep}
    print(f"Keeping classes: {args.keep} (ids: {sorted(list(keep_ids))})")

    # Detect available splits (train is required)
    candidate_splits = ["train", "valid", "val", "test"]
    splits = [sp for sp in candidate_splits if (root_in / sp / "images").exists()]
    if "val" in splits and "valid" in splits:
        # If both exist, prefer 'valid' and drop 'val' to avoid duplication
        splits.remove("val")
    if "train" not in splits:
        raise RuntimeError("Did not find train/images in the dataset")

    out_root = Path(str(root_in) + "_filtered")
    ensure_dirs(out_root, splits)

    totals = {}
    for sp in splits:
        kept, skipped = filter_split(root_in, out_root, sp, keep_ids)
        totals[sp] = (kept, skipped)

    # Write new data.yaml with only the kept class names (preserve original order)
    new_names = [n for n in names if n in args.keep]
    write_data_yaml(out_root, base_yaml, splits, new_names)

    print("\nSummary:")
    for sp,(k,s) in totals.items():
        print(f"  {sp}: kept {k} images, skipped {s}")
    print(f"\n✅ Filtered dataset written to: {out_root}")
    print(f"   New YAML: {out_root/'data.yaml'}")

if __name__ == "__main__":
    main()
