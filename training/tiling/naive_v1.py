import os
import cv2
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

def run_naive_tiling_multiclass(img_dir, label_dir, out_img_dir, out_label_dir, rows, cols):
    # 1. FORZA LA CREAZIONE DELLE CARTELLE a prescindere
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # 2. CERCA TUTTI I FORMATI POSSIBILI (Risolve il problema delle cartelle ignorate)
    supported_formats = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    images = []
    for fmt in supported_formats:
        images.extend(list(Path(img_dir).glob(fmt)))
        
    if not images:
        print(f"⚠️ Nessuna immagine trovata in {img_dir}! Controlla se la cartella è vuota.")
        return

    # Extract the split name (train or valid) dynamically from the directory path for the progress bar
    split_name = Path(img_dir).parent.name.upper()

    print(f"\n⏳ Inizio Tiling in: {img_dir}")
    print(f"Trovate {len(images)} immagini. Griglia impostata: {rows} righe x {cols} colonne")

    # =========================================================
    # 📊 TQDM PROGRESS BAR ADDED HERE
    # =========================================================
    for img_path in tqdm(images, desc=f"Elaborazione {split_name}", unit="img"):
        img_name = img_path.stem
        label_path = Path(label_dir) / f"{img_name}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None: 
            # Use tqdm.write so the warning doesn't break the progress bar visually
            tqdm.write(f"⚠️ Attenzione: Impossibile leggere l'immagine '{img_name}'. Salto.")
            continue
            
        img_h, img_w = img.shape[:2]
        
        tile_w = int(img_w / cols)
        tile_h = int(img_h / rows)
        
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    c, xc, yc, w, h = map(float, line.strip().split())
                    xmin, ymin, xmax, ymax = yolo_to_abs(xc, yc, w, h, img_w, img_h)
                    boxes.append({
                        'class': int(c),
                        'xmin': xmin, 'ymin': ymin, 
                        'xmax': xmax, 'ymax': ymax, 'area': (xmax-xmin)*(ymax-ymin)
                    })

        for row in range(rows):
            for col in range(cols):
                x_start = col * tile_w
                y_start = row * tile_h
                x_end = x_start + tile_w
                y_end = y_start + tile_h
                
                tile_boxes = []
                
                for b in boxes:
                    ix_min, iy_min = max(b['xmin'], x_start), max(b['ymin'], y_start)
                    ix_max, iy_max = min(b['xmax'], x_end), min(b['ymax'], y_end)
                    
                    if ix_min < ix_max and iy_min < iy_max:
                        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                        if (inter_area / b['area']) >= 0.4: 
                            loc_xmin, loc_ymin = max(0, ix_min - x_start), max(0, iy_min - y_start)
                            loc_xmax, loc_ymax = min(tile_w, ix_max - x_start), min(tile_h, iy_max - y_start)
                            n_xc, n_yc, n_w, n_h = abs_to_yolo(loc_xmin, loc_ymin, loc_xmax, loc_ymax, tile_w, tile_h)
                            
                            tile_boxes.append(f"{b['class']} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")
                
                if tile_boxes:
                    tile_name = f"{img_name}_r{row}_c{col}"
                    cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), img[y_start:y_end, x_start:x_end])
                    with open(os.path.join(out_label_dir, f"{tile_name}.txt"), 'w') as f:
                        f.write('\n'.join(tile_boxes))

    print(f"✅ Tiling completato per la cartella: {out_img_dir}\n")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Naive fixed-grid tiling (rows x cols); tiles must be multiples of 32."
    )
    p.add_argument("--base-dir", required=True,
                   help="Split root containing <split>/images and <split>/labels")
    p.add_argument("--out-dir", required=True, help="Destination root for tiled data")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    a = p.parse_args()

    for split in a.splits:
        run_naive_tiling_multiclass(
            f"{a.base_dir}/{split}/images", f"{a.base_dir}/{split}/labels",
            f"{a.out_dir}/{split}/images", f"{a.out_dir}/{split}/labels",
            rows=a.rows, cols=a.cols,
        )
