import os
import sys
import glob
import json
import shutil
import zipfile
import argparse
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = ['obstacle', 'right', 'no_right', 'straight', 'left', 'no_left']

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO Segmentation dataset to TriLiteNet format")
    parser.add_argument("--source", type=str, default="", help="Path to Roboflow dataset folder or zip file (leave empty for auto-detect)")
    parser.add_argument("--target", type=str, default="", help="Target directory for converted dataset (leave empty for auto-detect)")
    parser.add_argument("--update-config", action="store_true", default=True, help="Update lib/config/default.py with new dataset path")
    return parser.parse_args()

def auto_detect_source():
    """Find dataset folder or zip file in Kaggle input or current workspace."""
    candidates = []
    
    # 1. Search for any Roboflow COCO annotations in /kaggle/input or local
    search_dirs = [
        "/kaggle/input/**/_annotations.coco.json",
        "./**/_annotations.coco.json",
        "../**/_annotations.coco.json"
    ]
    for pattern in search_dirs:
        for ann_file in glob.glob(pattern, recursive=True):
            p = Path(ann_file).parent
            root = p.parent if p.name in ['train', 'valid', 'val', 'test'] else p
            if root not in candidates:
                candidates.append(root)

    # 2. Search for any folder with a 'train' subdirectory in /kaggle/input
    for train_dir in glob.glob("/kaggle/input/**/train", recursive=True):
        p = Path(train_dir).parent
        if p not in candidates and p != Path("/kaggle/input"):
            candidates.append(p)

    # 3. Direct subdirectories in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for sub in os.listdir("/kaggle/input"):
            full_p = Path("/kaggle/input") / sub
            if full_p.is_dir() and full_p not in candidates:
                if (full_p / "train").exists():
                    candidates.insert(0, full_p)
                else:
                    candidates.append(full_p)

    # 4. Search for any uploaded zip files
    for pattern in ["/kaggle/input/**/*.zip", "./*.zip", "../*.zip"]:
        for z in glob.glob(pattern, recursive=True):
            z_name = os.path.basename(z).lower()
            if 'trilite' not in z_name and 'checkpoint' not in z_name:
                zp = Path(z)
                if zp not in candidates:
                    candidates.append(zp)

    if candidates:
        return str(candidates[0])
    return None

def convert_coco_split(split_dir, out_base_dir, split_name, target_split="train"):
    """Convert a single split from Roboflow COCO to TriLiteNet (BDD-style) format."""
    coco_json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(coco_json_path):
        return {}

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in coco.get('categories', [])}
    valid_cats = {k: v for k, v in cat_map.items() if 'tset' not in v.lower() and 'test' not in v.lower()}
    print(f"\n[*] Split '{split_name}' -> Target '{target_split}'")
    print(f"    Raw categories: {cat_map}")
    if len(valid_cats) != len(cat_map):
        print(f"    [!] Ignored invalid/test classes: {[v for v in cat_map.values() if v not in valid_cats.values()]}")
    print(f"    Active categories: {list(valid_cats.values())}")

    img_id_to_anns = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        img_id_to_anns.setdefault(img_id, []).append(ann)

    img_out_dir = os.path.join(out_base_dir, 'images', target_split)
    det_out_dir = os.path.join(out_base_dir, 'det_annotations', target_split)
    da_out_dir = os.path.join(out_base_dir, 'da_seg_annotations', target_split)
    ll_out_dir = os.path.join(out_base_dir, 'll_seg_annotations', target_split)

    for d in [img_out_dir, det_out_dir, da_out_dir, ll_out_dir]:
        os.makedirs(d, exist_ok=True)

    images = coco.get('images', [])
    print(f"    Converting {len(images)} images...")

    stats = {
        'images': 0,
        'da_masks': 0,
        'boxes': {c: 0 for c in CLASS_NAMES}
    }

    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        width = int(img_info['width'])
        height = int(img_info['height'])
        stem = Path(file_name).stem

        src_img_path = os.path.join(split_dir, file_name)
        if not os.path.exists(src_img_path):
            continue

        # 1. Copy image
        dst_img_path = os.path.join(img_out_dir, f"{stem}.jpg")
        shutil.copy2(src_img_path, dst_img_path)
        stats['images'] += 1

        # 2. Render Drivable Area mask (da_seg)
        da_mask = np.zeros((height, width), dtype=np.uint8)
        has_da = False
        ll_mask = np.zeros((height, width), dtype=np.uint8)

        bdd_objects = []
        anns = img_id_to_anns.get(img_id, [])
        for ann in anns:
            cat_id = ann.get('category_id')
            cat_name = cat_map.get(cat_id, "").lower().strip()

            # Skip dummy / corrupted / mistaken class 'tset' or 'test'
            if 'tset' in cat_name or 'test' in cat_name:
                continue

            # Drivable road surface polygon
            if any(k in cat_name for k in ['lane', 'road', 'da']):
                segmentation = ann.get('segmentation', [])
                for seg in segmentation:
                    if len(seg) >= 6:
                        poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(da_mask, [poly], color=255)
                        has_da = True

            # Object detection bounding box (skip road surface classes)
            bbox = ann.get('bbox')
            if bbox and len(bbox) == 4 and not any(k in cat_name for k in ['lane', 'road', 'da']):
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                x1 = max(0.0, min(float(width), float(x)))
                y1 = max(0.0, min(float(height), float(y)))
                x2 = max(0.0, min(float(width), float(x + w)))
                y2 = max(0.0, min(float(height), float(y + h)))
                if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                    continue

                # 6 Classes Mapping
                if 'no right' in cat_name or 'no_right' in cat_name:
                    det_cat = 'no_right'
                elif 'no left' in cat_name or 'no_left' in cat_name:
                    det_cat = 'no_left'
                elif 'right' in cat_name:
                    det_cat = 'right'
                elif 'straight' in cat_name:
                    det_cat = 'straight'
                elif 'left' in cat_name:
                    det_cat = 'left'
                elif any(o in cat_name for o in ['obs', 'car', 'truck', 'bus', 'box', 'barrier', 'vehicle', 'vatcan']):
                    det_cat = 'obstacle'
                else:
                    # Ignore any other unrecognized labels
                    continue

                stats['boxes'][det_cat] += 1
                bdd_objects.append({
                    "category": det_cat,
                    "box2d": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    },
                    "raw_name": cat_name
                })

        if has_da:
            stats['da_masks'] += 1

        cv2.imwrite(os.path.join(da_out_dir, f"{stem}.png"), da_mask)
        cv2.imwrite(os.path.join(ll_out_dir, f"{stem}.png"), ll_mask)

        with open(os.path.join(det_out_dir, f"{stem}.json"), 'w', encoding='utf-8') as f:
            json.dump({"name": f"{stem}.jpg", "frames": [{"objects": bdd_objects}]}, f, indent=2)

    return stats

def update_default_config(dataset_dir, repo_dir):
    """Automatically update lib/config/default.py with the absolute or relative path to the converted dataset."""
    cfg_file = os.path.join(repo_dir, "lib", "config", "default.py")
    if not os.path.exists(cfg_file):
        return

    norm_path = os.path.abspath(dataset_dir).replace('\\', '/')
    with open(cfg_file, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace("_C.DATASET.DATAROOT = '../bdd100k/images'", f"_C.DATASET.DATAROOT = '{norm_path}/images'")
    content = content.replace("_C.DATASET.LABELROOT = '../bdd100k/det_annotations'", f"_C.DATASET.LABELROOT = '{norm_path}/det_annotations'")
    content = content.replace("_C.DATASET.MASKROOT = '../bdd100k/da_seg_annotations'", f"_C.DATASET.MASKROOT = '{norm_path}/da_seg_annotations'")
    content = content.replace("_C.DATASET.LANEROOT = '../bdd100k/ll_seg_annotations'", f"_C.DATASET.LANEROOT = '{norm_path}/ll_seg_annotations'")

    with open(cfg_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[+] Updated default.py DATAROOT -> {norm_path}")

def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent

    source_path = None
    if args.source:
        if os.path.exists(args.source):
            source_path = args.source
        else:
            print(f"[-] Warning: Specified source '{args.source}' was not found directly.")
            # Search for partial match in /kaggle/input
            term = os.path.basename(args.source.rstrip('/\\'))
            matches = glob.glob(f"/kaggle/input/**/*{term}*", recursive=True)
            if matches:
                source_path = matches[0]
                print(f"[+] Found match: {source_path}")

    if not source_path:
        source_path = auto_detect_source()

    if not source_path or not os.path.exists(source_path):
        print("[-] Error: No dataset source found.")
        if os.path.exists("/kaggle/input"):
            print(f"[*] Available items in /kaggle/input:")
            for item in os.listdir("/kaggle/input"):
                full_item = os.path.join("/kaggle/input", item)
                print(f"    - {item} ({'DIR' if os.path.isdir(full_item) else 'FILE'})")
        sys.exit(1)

    # Determine target directory
    if args.target:
        out_dir = Path(args.target)
    elif os.path.exists("/kaggle/working"):
        out_dir = Path("/kaggle/working/dataset_ucr")
    else:
        out_dir = repo_dir.parent / "dataset_ucr"

    print("==================================================")
    print("AUTO DATASET CONVERTER (ROBOFLOW COCO -> TRILITENET)")
    print(f"Source: {source_path}")
    print(f"Target: {out_dir}")
    print(f"Classes: {CLASS_NAMES}")
    print("==================================================")

    temp_extract = None
    if zipfile.is_zipfile(source_path):
        print(f"[*] Detected ZIP archive. Unzipping {source_path}...")
        temp_extract = repo_dir / "_temp_roboflow"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        with zipfile.ZipFile(source_path, 'r') as z:
            z.extractall(temp_extract)
        dataset_root = temp_extract
    else:
        dataset_root = Path(source_path)

    # If dataset_root has a single child directory that contains train/
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name != '_temp_roboflow']
    if not (dataset_root / "train").exists() and len(subdirs) == 1 and (subdirs[0] / "train").exists():
        dataset_root = subdirs[0]

    # Clean existing out_dir
    if out_dir.exists():
        print(f"[*] Cleaning old {out_dir}...")
        shutil.rmtree(out_dir)

    total_stats = {'train': {}, 'val': {}}

    # 1. Process Train
    train_dir = dataset_root / "train"
    if train_dir.exists():
        total_stats['train'] = convert_coco_split(str(train_dir), str(out_dir), "train", target_split="train")

    # 2. Process Valid
    valid_dir = dataset_root / "valid"
    if valid_dir.exists():
        total_stats['val'] = convert_coco_split(str(valid_dir), str(out_dir), "valid", target_split="val")

    # 3. Process Test (merged into val)
    test_dir = dataset_root / "test"
    if test_dir.exists():
        test_stats = convert_coco_split(str(test_dir), str(out_dir), "test", target_split="val")
        if total_stats['val']:
            total_stats['val']['images'] += test_stats.get('images', 0)
            total_stats['val']['da_masks'] += test_stats.get('da_masks', 0)
            for k in CLASS_NAMES:
                total_stats['val']['boxes'][k] += test_stats.get('boxes', {}).get(k, 0)
        else:
            total_stats['val'] = test_stats

    # Cleanup temp
    if temp_extract and temp_extract.exists():
        shutil.rmtree(temp_extract)

    print("\n==================================================")
    print("[+] SUMMARY STATISTICS:")
    print("--------------------------------------------------")
    for split_k in ['train', 'val']:
        st = total_stats.get(split_k, {})
        print(f"Split [{split_k.upper()}]:")
        print(f"  - Images: {st.get('images', 0)}")
        print(f"  - Drivable surface masks: {st.get('da_masks', 0)}")
        print(f"  - Bounding boxes per class:")
        for c, count in st.get('boxes', {}).items():
            print(f"      * {c:10s}: {count:3d}")
    print("==================================================")

    if args.update_config:
        update_default_config(str(out_dir), str(repo_dir))

    print(f"\n[+] Dataset ready for training at: {out_dir}")

if __name__ == "__main__":
    main()
