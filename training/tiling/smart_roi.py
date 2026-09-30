import os
import cv2
import yaml
import csv
import random
from pathlib import Path
from tqdm import tqdm


def yolo_to_abs(xc, yc, w, h, img_w, img_h):
    xmin = (xc - w / 2) * img_w
    ymin = (yc - h / 2) * img_h
    xmax = (xc + w / 2) * img_w
    ymax = (yc + h / 2) * img_h
    return xmin, ymin, xmax, ymax

def abs_to_yolo(xmin, ymin, xmax, ymax, tile_w, tile_h):
    w = (xmax - xmin) / tile_w
    h = (ymax - ymin) / tile_h
    xc = (xmin + xmax) / 2 / tile_w
    yc = (ymin + ymax) / 2 / tile_h
    return xc, yc, w, h

def run_roi_tiling(img_dir, label_dir, out_img_dir, out_label_dir, tile_size=640, overlap=0.10, keep_bg_ratio=0.05, split_name="train"):
    """
    Executes the tiling process and generates a YAML map for remapping/tracking,
    plus a CSV log tracking tile counts per original image.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # 1. Initialize the YAML Map / Log Dictionary
    yaml_map = {
        "parameters": {
            "tile_size_n": tile_size,
            "tile_size_m": tile_size,
            "overlap_o": overlap,
            "bg_threshold_t": keep_bg_ratio
        },
        "images": {}
    }

    supported_formats = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    images = []
    for fmt in supported_formats:
        images.extend(list(Path(img_dir).glob(fmt)))
        
    if not images:
        print(f"⚠️ No images found in {img_dir}!")
        return

    print(f"\n⏳ Starting ROI Tiling for {split_name.upper()}...")
    print(f"Parameters -> Size: {tile_size}x{tile_size}, Overlap: {overlap*100}%, BG Threshold: {keep_bg_ratio*100}%")

    tiles_with_objects = 0
    bg_tiles_kept = 0
    
    # CSV Log List
    generation_log = []

    # =========================================================
    # 📊 TQDM PROGRESS BAR ADDED HERE
    # =========================================================
    for img_path in tqdm(images, desc=f"Processing {split_name.upper()} images", unit="img"):
        img_name = img_path.stem
        label_path = Path(label_dir) / f"{img_name}.txt"
        
        # Check Img
        img = cv2.imread(str(img_path))
        if img is None: 
            # Use tqdm.write instead of print to avoid breaking the progress bar visually
            tqdm.write(f"⚠️ Warning: Could not read image '{img_name}'. Skipping.")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Initialize the image entry in the YAML map
        yaml_map["images"][img_name] = {
            "original_width": img_w,
            "original_height": img_h,
            "tiles": []
        }
        
        # Check Labels
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c, xc, yc, w, h = map(float, parts[:5])
                        xmin, ymin, xmax, ymax = yolo_to_abs(xc, yc, w, h, img_w, img_h)
                        boxes.append({
                            'class': int(c),
                            'xmin': xmin, 'ymin': ymin, 
                            'xmax': xmax, 'ymax': ymax, 'area': (xmax-xmin)*(ymax-ymin)
                        })

        # Calculate stride for overlap "o"
        stride = int(tile_size * (1 - overlap))
        x_starts = list(range(0, img_w - tile_size + 1, stride))
        y_starts = list(range(0, img_h - tile_size + 1, stride))
        
        if not x_starts or x_starts[-1] + tile_size < img_w: x_starts.append(max(0, img_w - tile_size))
        if not y_starts or y_starts[-1] + tile_size < img_h: y_starts.append(max(0, img_h - tile_size))

        tiles_from_this_image = 0

        for y_start in y_starts:
            for x_start in x_starts:
                x_end = min(x_start + tile_size, img_w)
                y_end = min(y_start + tile_size, img_h)
                
                cur_w, cur_h = x_end - x_start, y_end - y_start
                tile_boxes = []
                
                for b in boxes:
                    ix_min, iy_min = max(b['xmin'], x_start), max(b['ymin'], y_start)
                    ix_max, iy_max = min(b['xmax'], x_end), min(b['ymax'], y_end)
                    
                    if ix_min < ix_max and iy_min < iy_max:
                        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                        if (inter_area / b['area']) >= 0.3: 
                            loc_xmin, loc_ymin = max(0, ix_min - x_start), max(0, iy_min - y_start)
                            loc_xmax, loc_ymax = min(cur_w, ix_max - x_start), min(cur_h, iy_max - y_start)
                            n_xc, n_yc, n_w, n_h = abs_to_yolo(loc_xmin, loc_ymin, loc_xmax, loc_ymax, cur_w, cur_h)
                            
                            tile_boxes.append(f"{b['class']} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")
                
                tile_name = f"{img_name}_x{x_start}_y{y_start}"
                saved = False
                contains_object = False
                
                # Cut on object labels
                if tile_boxes:
                    cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), img[y_start:y_end, x_start:x_end])
                    with open(os.path.join(out_label_dir, f"{tile_name}.txt"), 'w') as f:
                        f.write('\n'.join(tile_boxes))
                    tiles_with_objects += 1
                    saved = True
                    contains_object = True
                    
                # Cut part of background with "t" threshold
                elif random.random() < keep_bg_ratio:
                    cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), img[y_start:y_end, x_start:x_end])
                    open(os.path.join(out_label_dir, f"{tile_name}.txt"), 'w').close()
                    bg_tiles_kept += 1
                    saved = True
                    contains_object = False
                
                # If we saved this tile, log it!
                if saved:
                    yaml_map["images"][img_name]["tiles"].append({
                        "tile_name": f"{tile_name}.jpg",
                        "x_start": x_start,
                        "y_start": y_start,
                        "width": cur_w,
                        "height": cur_h,
                        "contains_object": contains_object
                    })
                    tiles_from_this_image += 1
                    
        # Update CSV log for this image
        generation_log.append({
            "original_image": img_name,
            "tiles_generated": tiles_from_this_image
        })

    # Output YAML map to the root of the output directory
    yaml_path = os.path.join(str(Path(out_img_dir).parent), f"remapping_log_{split_name.lower()}.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_map, yaml_file, sort_keys=False, default_flow_style=False)

    # =========================================================
    # 📝 WRITE CSV LOG
    # =========================================================
    csv_path = os.path.join(str(Path(out_img_dir).parent), f"tile_counts_{split_name.lower()}.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original_Image", "Tiles_Generated"]) # Header
        for log_entry in generation_log:
            writer.writerow([log_entry["original_image"], log_entry["tiles_generated"]])

    print(f"✅ Generated {tiles_with_objects} object tiles and {bg_tiles_kept} background tiles.")
    print(f"🗺️  Remapping YAML map saved to: {yaml_path}")
    print(f"📊 CSV Log saved to: {csv_path}\n")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Smart-ROI tiling with negative background mining + remapping log."
    )
    p.add_argument("--base-dir", required=True,
                   help="Split root containing <split>/images and <split>/labels")
    p.add_argument("--out-dir", required=True, help="Destination root for tiled data")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.10)
    p.add_argument("--keep-bg", type=float, default=0.05,
                   help="Fraction of empty tiles retained as negative samples")
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    a = p.parse_args()

    for split in a.splits:
        run_roi_tiling(
            f"{a.base_dir}/{split}/images", f"{a.base_dir}/{split}/labels",
            f"{a.out_dir}/{split}/images", f"{a.out_dir}/{split}/labels",
            tile_size=a.tile_size, overlap=a.overlap,
            keep_bg_ratio=a.keep_bg, split_name=split,
        )
