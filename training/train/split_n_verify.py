import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm


# =========================================================
# 🧬 UTILITY: FILENAME PARSING
# =========================================================
def parse_filename(filename):
    """
    Extracts the base sequence name and frame number from exported frame filenames (strips the `.rf.<hash>` export suffix).
    """
    base = filename.split('.rf.')[0]
    base = base.replace('_jpg', '').replace('_JPG', '')
    
    match_frame = re.search(r'(.*_frame)(\d+)$', base)
    if match_frame:
        return match_frame.group(1), int(match_frame.group(2))
    
    match_photo = re.search(r'(.*-)(\d+)-?$', base)
    if match_photo:
        return match_photo.group(1), int(match_photo.group(2))
        
    return base, None

# =========================================================
# ⏱️ MODULE 1: TEMPORAL SPLITTING WITH BUFFER ZONES
# =========================================================
def create_temporal_splits(raw_img_dir, raw_lbl_dir, base_out_dir, test_pct=0.10, valid_pct=0.10, buffer_frames=2):
    """
    Sorts each sequence chronologically and splits it (Test -> Valid -> Train).
    Drops 'buffer_frames' at the boundaries to prevent data leakage.
    """
    raw_images = list(Path(raw_img_dir).glob("*.[jJ][pP][gG]"))
    if not raw_images:
        print(f"❌ No images found in {raw_img_dir}! Please check the path.")
        return False

    print(f"\n⏱️ Grouping and temporally sorting {len(raw_images)} raw images...")
    
    # 1. Group images by sequence and extract their frame numbers for sorting
    sequences = {}
    for img_path in raw_images:
        seq_base, frame_num = parse_filename(img_path.name)
        if seq_base not in sequences:
            sequences[seq_base] = []
        
        # If frame_num is missing for some reason, default to 0
        sort_idx = frame_num if frame_num is not None else 0
        sequences[seq_base].append((sort_idx, img_path))

    # 2. Sort each sequence chronologically
    for seq_base in sequences:
        sequences[seq_base].sort(key=lambda x: x[0])

    print(f"✅ Found {len(sequences)} unique sequences.")

    # Setup directories
    splits = ["train", "valid", "test"]
    for s in splits:
        os.makedirs(Path(base_out_dir) / s / "images", exist_ok=True)
        os.makedirs(Path(base_out_dir) / s / "labels", exist_ok=True)

    counts = {"train": 0, "valid": 0, "test": 0, "dropped": 0}
    
    # 3. Execute the Chronological Splitting
    for seq_base, items in tqdm(sequences.items(), desc="🚚 Copying Temporal Splits", unit="seq"):
        total_frames = len(items)
        
        # Calculate the exact cutoff indices for this specific sequence
        test_end_idx = int(total_frames * test_pct)
        valid_end_idx = test_end_idx + int(total_frames * valid_pct)
        
        for i, (frame_num, img_path) in enumerate(items):
            target_split = None
            
            # --- ZONE 1: TEST (0 to 10%) ---
            if i < test_end_idx:
                # Drop frames at the end of the Test zone
                if i >= test_end_idx - buffer_frames:
                    counts["dropped"] += 1
                    continue 
                target_split = "test"
                
            # --- ZONE 2: VALID (10% to 20%) ---
            elif i < valid_end_idx:
                # Drop frames at the start and end of the Valid zone
                if i < test_end_idx + buffer_frames or i >= valid_end_idx - buffer_frames:
                    counts["dropped"] += 1
                    continue
                target_split = "valid"
                
            # --- ZONE 3: TRAIN (20% to 100%) ---
            else:
                # Drop frames at the start of the Train zone
                if i < valid_end_idx + buffer_frames:
                    counts["dropped"] += 1
                    continue
                target_split = "train"

            # Copy Image and Label
            if target_split:
                lbl_path = Path(raw_lbl_dir) / f"{img_path.stem}.txt"
                shutil.copy(img_path, Path(base_out_dir) / target_split / "images" / img_path.name)
                
                if lbl_path.exists():
                    shutil.copy(lbl_path, Path(base_out_dir) / target_split / "labels" / lbl_path.name)
                    
                counts[target_split] += 1

    print(f"\n🎉 Temporal Split Complete!")
    print(f"   Test (0-10%):   {counts['test']} images")
    print(f"   Valid (10-20%): {counts['valid']} images")
    print(f"   Train (20-100%):{counts['train']} images")
    print(f"   🛡️ Dropped (Buffer): {counts['dropped']} boundary frames to prevent leakage")
    return True

# =========================================================
# 🕵️ MODULE 2: LEAKAGE VERIFICATION
# =========================================================
def verify_no_leakage(base_out_dir, adjacency_threshold=1):
    print("\n🕵️ Running Data Leakage Security Check...")
    base_path = Path(base_out_dir)
    
    train_sequences = {}
    train_count = 0
    for img_path in (base_path / "train" / "images").glob("*.[jJ][pP][gG]"):
        train_count += 1
        seq_base, frame_num = parse_filename(img_path.name)
        if frame_num is not None:
            if seq_base not in train_sequences:
                train_sequences[seq_base] = set()
            train_sequences[seq_base].add(frame_num)

    if train_count == 0:
        print("⚠️ No images found in Train folder to check against.")
        return False

    leak_found = False
    
    for split in ["valid", "test"]:
        split_dir = base_path / split / "images"
        if not split_dir.exists(): continue
        
        for img_path in split_dir.glob("*.[jJ][pP][gG]"):
            seq_base, frame_num = parse_filename(img_path.name)
            
            if frame_num is not None and seq_base in train_sequences:
                train_frames = train_sequences[seq_base]
                
                for f in range(frame_num - adjacency_threshold, frame_num + adjacency_threshold + 1):
                    if f in train_frames:
                        if f == frame_num:
                            print(f"🚨 EXACT MATCH LEAK: '{img_path.name}' is in BOTH Train and {split.upper()}!")
                        else:
                            print(f"🚨 ADJACENCY LEAK: {split.upper()} frame '{img_path.name}' is too close to Train frame {f}")
                        leak_found = True
                        break
                        
    if not leak_found:
        print("✅ SECURITY PASSED: Zero data leakage detected between temporal splits.")
    return not leak_found

# =========================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Chronological (sequence-aware) train/valid/test split with leakage buffer + verification."
    )
    p.add_argument("--raw-dir", required=True,
                   help="Directory with raw 'images/' and 'labels/' subfolders")
    p.add_argument("--out-dir", required=True, help="Destination split root")
    p.add_argument("--test-pct", type=float, default=0.10)
    p.add_argument("--valid-pct", type=float, default=0.10)
    p.add_argument("--buffer-frames", type=int, default=2,
                   help="Frames dropped at each split boundary to prevent temporal leakage")
    a = p.parse_args()

    if not (Path(a.raw_dir) / "images").exists():
        raise SystemExit(f"\u274c Raw directory {a.raw_dir}/images not found.")

    ok = create_temporal_splits(
        raw_img_dir=f"{a.raw_dir}/images",
        raw_lbl_dir=f"{a.raw_dir}/labels",
        base_out_dir=a.out_dir,
        test_pct=a.test_pct,
        valid_pct=a.valid_pct,
        buffer_frames=a.buffer_frames,
    )
    if ok:
        verify_no_leakage(base_out_dir=a.out_dir, adjacency_threshold=1)
