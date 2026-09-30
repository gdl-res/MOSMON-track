import os
import glob
import cv2
from ultralytics import YOLO
import torch  # <--- Needed for VRAM management
import gc     # <--- Python's Garbage Collector

# Mode table (Modes 0-9; see the Training section of the README).
# id -> (description, dataset_yaml, exp_name, apply_correction, apply_copy_paste)
MODES = {
    0: ("Baseline (original 4K)",              "dataset_baseline.yaml",                "0_baseline",                       False, False),
    1: ("Naive grid tiling",                   "dataset_naive.yaml",                   "1_tiling_naive",                   False, False),
    2: ("Overlapping tiling",                  "dataset_overlapping.yaml",             "2_tiling_overlapping",             False, False),
    3: ("Smart-ROI tiling",                    "dataset_smart_roi.yaml",               "3_smart_roi",                      False, False),
    4: ("Baseline + dfl correction",           "dataset_baseline.yaml",                "4_baseline_corrected",             True,  False),
    5: ("Naive + dfl correction",              "dataset_naive.yaml",                   "5_tiling_naive_corrected",         True,  False),
    6: ("Overlapping + dfl correction",        "dataset_overlapping.yaml",             "6_tiling_overlapping_corrected",   True,  False),
    7: ("Smart-ROI + dfl correction",          "dataset_smart_roi.yaml",               "7_smart_roi_corrected",            True,  False),
    8: ("Overlapping oversampled",             "dataset_overlapping_oversampled.yaml", "8_tiling_overlapping_oversampled", False, False),
    9: ("Overlapping oversampled + copy-paste","dataset_overlapping_oversampled.yaml", "9_oo_copy_paste",                  False, True),
}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    default_cfg = str(Path(__file__).resolve().parent.parent / "configs")
    p = argparse.ArgumentParser(description="Train a YOLO11 model for one training mode (0-9).")
    p.add_argument("--mode", type=int, required=True, choices=sorted(MODES))
    p.add_argument("--configs-dir", default=default_cfg,
                   help="Directory holding the dataset_*.yaml files")
    p.add_argument("--project", required=True, help="Ultralytics runs/ output directory")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    desc, yaml_name, exp_name, apply_correction, apply_copy_paste = MODES[a.mode]
    data_yaml = str(Path(a.configs_dir) / yaml_name)
    print(f"\U0001f680 Training mode {a.mode}: {desc}")

    model = YOLO(a.weights)

    train_args = {
        "data": data_yaml,
        "epochs": a.epochs,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "device": a.device,
        "plots": False,
        "workers": 2,
        "project": a.project,
        "name": exp_name,
        "cache": "disk",
        "patience": 32,
        "save_period": 32,
        "freeze": 10,
        # Biological & water augmentations (constant across all modes)
        "cos_lr": True,
        "flipud": 0.0,
        "degrees": 0.0,
        "hsv_s": 0.9,
        "hsv_v": 0.6,
        # Advanced augmentations off unless a mode requests them
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    if apply_copy_paste:
        print("\u2702\ufe0f Injecting calibrated Mosaic & Copy-Paste augmentations")
        train_args["mosaic"] = 1.0
        train_args["copy_paste"] = 0.15
        train_args["erasing"] = 0.2
        train_args["scale"] = 0.1
        train_args["bgr"] = 0.1

    if apply_correction:
        print("\U0001f489 Injecting DFL loss correction (dfl=3.0) to fight class imbalance")
        train_args["box"] = 7.5
        train_args["cls"] = 0.5
        train_args["dfl"] = 3.0

    try:
        model.train(**train_args)
        print(f"\u2705 Training for mode {a.mode} ({desc}) completed")
    except Exception as e:
        print(f"\u274c ERROR during training: {e}")
    finally:
        print("\U0001f9f9 Flushing GPU memory...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
