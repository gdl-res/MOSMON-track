import os
import cv2
import csv
import yaml  # <--- AGGIUNTA LA LIBRERIA YAML
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

def run_naive_tiling_multiclass(img_dir, label_dir, out_img_dir, out_label_dir, tile_size=1280):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    supported_formats = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    images = []
    for fmt in supported_formats:
        images.extend(list(Path(img_dir).glob(fmt)))
        
    if not images:
        print(f"⚠️ Nessuna immagine trovata in {img_dir}! Controlla se la cartella è vuota.")
        return

    split_name = Path(img_dir).parent.name.upper()

    print(f"\n⏳ Inizio Tiling in: {img_dir}")
    print(f"Trovate {len(images)} immagini. Dimensione Tile fissa: {tile_size}x{tile_size} px")

    # CSV Log List
    generation_log = []
    
    # =========================================================
    # 🗺️ INIZIALIZZAZIONE DELLA MAPPA YAML
    # =========================================================
    yaml_map = {
        "parameters": {
            "tile_size_n": tile_size,
            "tile_size_m": tile_size,
            "type": "naive_fixed_size"
        },
        "images": {}
    }

    # =========================================================
    # 📊 TQDM PROGRESS BAR
    # =========================================================
    for img_path in tqdm(images, desc=f"Elaborazione {split_name}", unit="img"):
        img_name = img_path.stem
        label_path = Path(label_dir) / f"{img_name}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None: 
            tqdm.write(f"⚠️ Attenzione: Impossibile leggere l'immagine '{img_name}'. Salto.")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Inizializza l'immagine nel log YAML salvando le dimensioni ORIGINALI
        yaml_map["images"][img_name] = {
            "original_width": img_w,
            "original_height": img_h,
            "tiles": []
        }
        
        # --- AUTO-PADDING PER IMMAGINI PICCOLE ---
        # Se l'immagine è più piccola di 1280, aggiunge bordi neri per non far crashare YOLO
        pad_bottom = max(0, tile_size - img_h)
        pad_right = max(0, tile_size - img_w)
        if pad_bottom > 0 or pad_right > 0:
            img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img_h, img_w = img.shape[:2] # Aggiorna le dimensioni post-padding
        
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

        # --- CALCOLO GENERICO DELLE FINESTRE ---
        x_starts = list(range(0, img_w, tile_size))
        y_starts = list(range(0, img_h, tile_size))
        
        # Assicura che l'ultima finestra non esca fuori dai bordi (garantisce tile esattamente 1280x1280)
        for i in range(len(x_starts)):
            if x_starts[i] + tile_size > img_w:
                x_starts[i] = img_w - tile_size
        for i in range(len(y_starts)):
            if y_starts[i] + tile_size > img_h:
                y_starts[i] = img_h - tile_size
                
        # Rimuove eventuali doppioni causati dall'allineamento ai bordi
        x_starts = sorted(list(set(x_starts)))
        y_starts = sorted(list(set(y_starts)))

        tiles_from_this_image = 0

        for y_start in y_starts:
            for x_start in x_starts:
                x_end = x_start + tile_size
                y_end = y_start + tile_size
                
                tile_boxes = []
                
                for b in boxes:
                    ix_min, iy_min = max(b['xmin'], x_start), max(b['ymin'], y_start)
                    ix_max, iy_max = min(b['xmax'], x_end), min(b['ymax'], y_end)
                    
                    if ix_min < ix_max and iy_min < iy_max:
                        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                        # Se almeno il 40% dell'oggetto è nel tile
                        if (inter_area / b['area']) >= 0.4: 
                            loc_xmin, loc_ymin = max(0, ix_min - x_start), max(0, iy_min - y_start)
                            loc_xmax, loc_ymax = min(tile_size, ix_max - x_start), min(tile_size, iy_max - y_start)
                            n_xc, n_yc, n_w, n_h = abs_to_yolo(loc_xmin, loc_ymin, loc_xmax, loc_ymax, tile_size, tile_size)
                            
                            tile_boxes.append(f"{b['class']} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")
                
                # Salva solo se ci sono box
                if tile_boxes:
                    tile_name = f"{img_name}_x{x_start}_y{y_start}"
                    cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), img[y_start:y_end, x_start:x_end])
                    with open(os.path.join(out_label_dir, f"{tile_name}.txt"), 'w') as f:
                        f.write('\n'.join(tile_boxes))
                    tiles_from_this_image += 1
                    
                    # Aggiunge il tile al log YAML per il tracking
                    yaml_map["images"][img_name]["tiles"].append({
                        "tile_name": f"{tile_name}.jpg",
                        "x_start": x_start,
                        "y_start": y_start,
                        "width": tile_size,
                        "height": tile_size,
                        "contains_object": True # In questo script salviamo solo se ci sono box
                    })

        # Aggiorna il log CSV per questa immagine
        generation_log.append({
            "original_image": img_name,
            "tiles_generated": tiles_from_this_image
        })

    # =========================================================
    # 📝 SCRITTURA DEL LOG YAML (REMAPPING)
    # =========================================================
    yaml_path = os.path.join(str(Path(out_img_dir).parent), f"remapping_log_{split_name.lower()}.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_map, yaml_file, sort_keys=False, default_flow_style=False)

    # =========================================================
    # 📝 SCRITTURA DEL LOG CSV (COUNT)
    # =========================================================
    csv_path = os.path.join(str(Path(out_img_dir).parent), f"tile_counts_{split_name.lower()}.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original_Image", "Tiles_Generated"]) # Header
        for log_entry in generation_log:
            writer.writerow([log_entry["original_image"], log_entry["tiles_generated"]])

    print(f"✅ Tiling completato per: {out_img_dir}")
    print(f"🗺️  Remapping YAML map salvato in: {yaml_path}")
    print(f"📊 CSV Log salvato in: {csv_path}\n")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Naive fixed-size tiling (absolute tile size, auto-padded)."
    )
    p.add_argument("--base-dir", required=True,
                   help="Split root containing <split>/images and <split>/labels")
    p.add_argument("--out-dir", required=True, help="Destination root for tiled data")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    a = p.parse_args()

    for split in a.splits:
        run_naive_tiling_multiclass(
            f"{a.base_dir}/{split}/images", f"{a.base_dir}/{split}/labels",
            f"{a.out_dir}/{split}/images", f"{a.out_dir}/{split}/labels",
            tile_size=a.tile_size,
        )
