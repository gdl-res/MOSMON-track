import os
import glob
import cv2
from ultralytics import YOLO


def get_auto_imgsz(dataset_folder):
    """
    Looks for the first image in the dataset and returns its size.
    YOLO requires the size to be a multiple of 32.
    """
    search_path = os.path.join(dataset_folder, "train", "images", "*.jpg")
    images = glob.glob(search_path) + glob.glob(os.path.join(dataset_folder, "train", "images", "*.JPG"))
    
    if not images:
        print(f"⚠️ Warning: No images found in {search_path}. Using default 640.")
        return 640
        
    first_image = cv2.imread(images[0])
    h, w = first_image.shape[:2]
    auto_size = max(h, w)
    
    # Mathematical correction: force multiple of 32 if it isn't already
    if auto_size % 32 != 0:
        auto_size = ((auto_size // 32) + 1) * 32
        
    print(f"📏 Auto-detected image size: {auto_size}x{auto_size} px")
    return auto_size

# Mode table (see training/README.md). id -> (description, dataset_yaml, dataset_subdir, exp_name, apply_correction)
MODES = {
    0: ("Baseline (original 4K)",        "dataset_baseline.yaml",    "baseline",  "0_baseline",                     False),
    1: ("Naive grid tiling",             "dataset_naive.yaml",       "naive",     "1_tiling_naive",                 False),
    2: ("Overlapping tiling",            "dataset_overlapping.yaml", "overlap",   "2_tiling_overlapping",           False),
    3: ("Smart-ROI tiling",              "dataset_smart_roi.yaml",   "smart_roi", "3_smart_roi",                    True),
    4: ("Baseline + dfl correction",     "dataset_baseline.yaml",    "baseline",  "4_baseline_corrected",           True),
    5: ("Naive + dfl correction",        "dataset_naive.yaml",       "naive",     "5_tiling_naive_corrected",       True),
    6: ("Overlapping + dfl correction",  "dataset_overlapping.yaml", "overlap",   "6_tiling_overlapping_corrected", True),
    7: ("Smart-ROI + dfl correction",    "dataset_smart_roi.yaml",   "smart_roi", "7_smart_roi_corrected",          True),
}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    default_cfg = str(Path(__file__).resolve().parent.parent / "configs")
    p = argparse.ArgumentParser(
        description="Sequentially train YOLO11 for every training mode with auto image-size detection."
    )
    p.add_argument("--datasets-root", required=True,
                   help="Root holding one tiled dataset folder per mode (baseline/, naive/, ...)")
    p.add_argument("--configs-dir", default=default_cfg,
                   help="Directory holding the dataset_*.yaml files")
    p.add_argument("--project", required=True, help="Ultralytics runs/ output directory")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="0")
    p.add_argument("--modes", nargs="+", type=int, default=sorted(MODES),
                   help="Subset of modes to train (default: all)")
    a = p.parse_args()

    print(f"\U0001f680 Scheduled to train {len(a.modes)} models sequentially.")
    for mode_id in a.modes:
        desc, yaml_name, subdir, exp_name, apply_correction = MODES[mode_id]
        print("=" * 60)
        print(f"\U0001f52c Mode {mode_id}: {desc}")
        try:
            model = YOLO(a.weights)
            dataset_folder = f"{a.datasets_root}/{subdir}"
            dynamic_imgsz = get_auto_imgsz(dataset_folder)

            train_args = {
                "data": str(Path(a.configs_dir) / yaml_name),
                "epochs": a.epochs,
                "imgsz": dynamic_imgsz,
                "batch": 8,
                "device": a.device,
                "plots": True,
                "workers": 2,
                "project": a.project,
                "name": exp_name,
                "cache": False,
                "patience": 25,
            }
            if apply_correction:
                print("\U0001f489 Injecting DFL loss correction (dfl=3.0)")
                train_args["box"] = 7.5
                train_args["cls"] = 0.5
                train_args["dfl"] = 3.0

            model.train(**train_args)
            print(f"\u2705 Mode {mode_id} ({desc}) completed")
        except Exception as e:
            print(f"\u274c ERROR training mode {mode_id}: {e}\n\u23ed Moving on...")

    print("\U0001f389 All scheduled training runs finished.")
