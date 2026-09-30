"""Plot a training metric over epochs for every mode on one figure."""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# label -> experiment folder name (see training/README.md)
RUNS = {
    "0. Baseline": "0_baseline",
    "1. Naive Tiling": "1_tiling_naive",
    "2. Overlap Tiling": "2_tiling_overlapping",
    "3. ROI Tiling": "3_smart_roi",
    "4. Baseline + Correction": "4_baseline_corrected",
    "5. Naive + Correction": "5_tiling_naive_corrected",
    "6. Overlap + Correction": "6_tiling_overlapping_corrected",
    "7. ROI + Correction": "7_smart_roi_corrected",
}

STYLES = {
    "0. Baseline": {"color": "#1f77b4", "linestyle": "-"},
    "1. Naive Tiling": {"color": "#ff7f0e", "linestyle": "-"},
    "2. Overlap Tiling": {"color": "#2ca02c", "linestyle": "-"},
    "3. ROI Tiling": {"color": "#d62728", "linestyle": "-"},
    "4. Baseline + Correction": {"color": "#1f77b4", "linestyle": "--"},
    "5. Naive + Correction": {"color": "#ff7f0e", "linestyle": "--"},
    "6. Overlap + Correction": {"color": "#2ca02c", "linestyle": "--"},
    "7. ROI + Correction": {"color": "#d62728", "linestyle": "--"},
}


def plot_comparison(runs_dir, metric, out_dir):
    plt.figure(figsize=(12, 7))
    data_found = False

    for label, folder in RUNS.items():
        csv_path = os.path.join(runs_dir, folder, "results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing {csv_path}; skipping '{label}'.")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not in {csv_path}.")
            continue
        data_found = True
        plt.plot(df["epoch"], df[metric], label=label, linewidth=2.5,
                 marker="o", markersize=4, **STYLES[label])

    if not data_found:
        print("❌ No results.csv files found.")
        return

    metric_name = metric.split("/")[-1]
    plt.title(f"YOLO model comparison: {metric_name} over epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(f"Accuracy ({metric_name})", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.02, 0.5))

    safe = metric_name.replace("/", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(out_dir, f"model_comparison_{safe}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay a training metric across modes.")
    p.add_argument("--runs-dir", required=True, help="Ultralytics project dir holding <mode>/results.csv")
    p.add_argument("--out-dir", default=".", help="Where to save the PNGs")
    p.add_argument("--metrics", nargs="+", default=["metrics/mAP50(B)", "train/box_loss"])
    a = p.parse_args()

    print("\U0001f4ca Generating comparison graphs...")
    for metric in a.metrics:
        plot_comparison(a.runs_dir, metric, a.out_dir)
