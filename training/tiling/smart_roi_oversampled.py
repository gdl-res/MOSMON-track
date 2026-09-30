import os
import cv2
import yaml
import csv
import random
from pathlib import Path
from tqdm import tqdm


# ==========================================
# ⚙️ MATHEMATICAL BALANCING PARAMETERS
# ==========================================
# We map each YOLO Class ID to its exact required overlap to reach ~118,000 instances
CLASS_OVERLAPS = {
    1: 0.10,  # Aedes albopictus   (Target: 1x    -> 10% overlap)
    0: 0.50,  # Aedes aegypti      (Target: 3.27x -> 50% overlap)
    3: 0.60,  # Culex pipiens      (Target: 5.17x -> 60% overlap)
    2: 0.68   # Anopheles stephensi(Target: 7.70x -> 68% overlap)
}

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

def run_roi_oversampled_tiling(img_dir, label_dir, out_img_dir, out_label_dir, tile_size=640, base_overlap=0.10, keep_bg_ratio=0.05, split_name="train"):
    """
    Executes the adaptive oversampling tiling process with negative mining.
    Generates a YAML map for remapping/tracking, plus a CSV log.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # 1. Initialize the YAML Map / Log Dictionary
    yaml_map = {
        "parameters": {
            "tile_size_n": tile_size,
            "tile_size_m": tile_size,
            "base_overlap": base_overlap,
            "bg_threshold_t": keep_bg_ratio,
            "type": "smart_roi_adaptive_oversampled"
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

    print(f"\n⏳ Starting Smart ROI Oversampled Tiling for {split_name.upper()}...")
    print(f"Parameters -> Size: {tile_size}x{tile_size}, Base Overlap: {int(base_overlap*100)}%, BG Threshold: {keep_bg_ratio*100}%")

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
            tqdm.write(f"⚠️ Warning: Could not read image '{img_name}'. Skipping.")
            continue
            
        img_h, img_w = img.shape[:2]
        
        yaml_map["images"][img_name] = {
            "original_width": img_w,
            "original_height": img_h,
            "tiles": []
        }
        
        # Check Labels
        boxes = []
        labeled_px_coords = []
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
                        # Log the exact center to trigger the adaptive slowdown
                        abs_xc = (xmin + xmax) / 2
                        abs_yc = (ymin + ymax) / 2
                        labeled_px_coords.append((int(c), abs_xc, abs_yc))

        # =========================================================
        # 🧮 DYNAMIC ADAPTIVE TILING LOGIC WITH ROI BACKGROUND FILTER
        # =========================================================
        tiles_from_this_image = 0
        y_start = 0

        while y_start < img_h:
            x_start = 0
            # Track the smallest Y-stride in this row so the next row overlaps properly
            next_y_stride = int(tile_size * (1 - base_overlap))

            while x_start < img_w:
                x_end = min(x_start + tile_size, img_w)
                y_end = min(y_start + tile_size, img_h)
                
                cur_w, cur_h = x_end - x_start, y_end - y_start
                
                # Check EVERY bug in the image to see if it falls inside this tile
                current_overlap = base_overlap
                for (cls_id, bx, by) in labeled_px_coords:
                    if x_start <= bx <= x_end and y_start <= by <= y_end:
                        bug_req_overlap = CLASS_OVERLAPS.get(cls_id, base_overlap)
                        if bug_req_overlap > current_overlap:
                            current_overlap = bug_req_overlap
                
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
                
                # Move to the next X column based on the adaptive overlap
                stride_x = max(10, int(tile_size * (1 - current_overlap)))
                x_start += stride_x
                
                # Update the Y stride for the next row if this tile was dense
                next_y_stride = min(next_y_stride, stride_x)

            y_start += next_y_stride
                    
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
        description="Adaptive oversampled Smart-ROI tiling with negative mining."
    )
    p.add_argument("--base-dir", required=True,
                   help="Split root containing <split>/images and <split>/labels")
    p.add_argument("--out-dir", required=True, help="Destination root for tiled data")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.10, help="Base overlap coefficient")
    p.add_argument("--keep-bg", type=float, default=0.05,
                   help="Fraction of empty tiles retained as negative samples")
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    a = p.parse_args()

    for split in a.splits:
        run_roi_oversampled_tiling(
            f"{a.base_dir}/{split}/images", f"{a.base_dir}/{split}/labels",
            f"{a.out_dir}/{split}/images", f"{a.out_dir}/{split}/labels",
            tile_size=a.tile_size, base_overlap=a.overlap,
            keep_bg_ratio=a.keep_bg, split_name=split,
        )
