"""Summarise every mode's best epoch from its Ultralytics results.csv into one table.

Reads <runs-dir>/<mode>/results.csv for each mode, picks the row with the highest
mAP50-95, and writes master_model_comparison.csv.
"""
import argparse
import os

import pandas as pd

# Experiment folder names produced by train.py (see training/README.md).
MODES = [
    "0_baseline",
    "1_tiling_naive",
    "2_tiling_overlapping",
    "3_smart_roi",
    "4_baseline_corrected",
    "5_tiling_naive_corrected",
    "6_tiling_overlapping_corrected",
    "7_smart_roi_corrected",
    "8_tiling_overlapping_oversampled",
    "9_oo_copy_paste",
]


def build_comparison(runs_dir, modes):
    rows = []
    print("\U0001f4ca Analyzing YOLO result CSVs...")
    for mode_name in modes:
        file_path = os.path.join(runs_dir, mode_name, "results.csv")
        if not os.path.exists(file_path):
            print(f"⚠️  Missing results.csv for {mode_name}")
            continue
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
            rows.append({
                "Mode": mode_name,
                "Best_Epoch": int(best["epoch"]),
                "mAP50-95": round(best["metrics/mAP50-95(B)"], 4),
                "mAP50": round(best["metrics/mAP50(B)"], 4),
                "Precision": round(best["metrics/precision(B)"], 4),
                "Recall": round(best["metrics/recall(B)"], 4),
                "Train_Box_Loss": round(best["train/box_loss"], 4),
                "Val_Box_Loss": round(best["val/box_loss"], 4),
                "Train_Cls_Loss": round(best["train/cls_loss"], 4),
                "Val_Cls_Loss": round(best["val/cls_loss"], 4),
            })
            print(f"✅ Processed {mode_name}")
        except Exception as e:
            print(f"❌ Error processing {mode_name}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate per-mode YOLO results into one comparison table.")
    p.add_argument("--runs-dir", required=True, help="Ultralytics project dir holding <mode>/results.csv")
    p.add_argument("--out", default=None, help="Output CSV path (default: <runs-dir>/master_model_comparison.csv)")
    a = p.parse_args()

    df = build_comparison(a.runs_dir, MODES)
    if df.empty:
        raise SystemExit("❌ No data generated. Check --runs-dir.")

    out = a.out or os.path.join(a.runs_dir, "master_model_comparison.csv")
    df.to_csv(out, index=False)
    print("\n\U0001f3c6 Final comparison table:")
    print(df.to_string(index=False))
    print(f"\n\U0001f4c1 Saved to: {out}")
