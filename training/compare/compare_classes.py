"""Per-class (per-species) mAP comparison across modes via Ultralytics val().

Loads each mode's best.pt, validates it against that mode's dataset YAML, and reports
mAP50 / mAP50-95 per species. Writes class_comparison_results.csv plus two bar charts.
"""
import argparse
import os

import pandas as pd
from ultralytics import YOLO

# experiment folder name -> dataset YAML used to validate it (see training/README.md)
MODES = {
    "0_baseline": "dataset_baseline.yaml",
    "1_tiling_naive": "dataset_naive.yaml",
    "2_tiling_overlapping": "dataset_overlapping.yaml",
    "3_smart_roi": "dataset_smart_roi.yaml",
    "4_baseline_corrected": "dataset_baseline_corrected.yaml",
    "5_tiling_naive_corrected": "dataset_naive_corrected.yaml",
    "6_tiling_overlapping_corrected": "dataset_overlapping_corrected.yaml",
    "7_smart_roi_corrected": "dataset_smart_roi_corrected.yaml",
    "8_tiling_overlapping_oversampled": "dataset_overlapping_oversampled.yaml",
}


def evaluate(runs_dir, configs_dir):
    results = []
    print("\U0001f680 Per-class evaluation...")
    for mode_name, yaml_name in MODES.items():
        weights = os.path.join(runs_dir, mode_name, "weights", "best.pt")
        data_yaml = os.path.join(configs_dir, yaml_name)
        if not os.path.exists(weights):
            print(f"⚠️  Missing weights for {mode_name}; skipping.")
            continue
        print(f"\U0001f50d Evaluating {mode_name}...")
        try:
            model = YOLO(weights)
            metrics = model.val(data=data_yaml, split="val", plots=False, verbose=False)
            names = metrics.names
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                results.append({
                    "Mode": mode_name,
                    "Class": names[class_idx],
                    "mAP50": round(metrics.box.ap50[i], 4),
                    "mAP50-95": round(metrics.box.ap[i], 4),
                })
        except Exception as e:
            print(f"❌ Error evaluating {mode_name}: {e}")
    return pd.DataFrame(results)


def make_plot(df, metric_col, title, save_path):
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    plt = sns.mpl.pyplot
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="Mode", y=metric_col, hue="Class", edgecolor=".2")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Image processing strategy (mode)", fontsize=12, fontweight="bold")
    plt.ylabel(f"Accuracy ({metric_col})", fontsize=12, fontweight="bold")
    plt.xticks(rotation=15, fontsize=10)
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot {save_path}")


if __name__ == "__main__":
    from pathlib import Path

    default_cfg = str(Path(__file__).resolve().parent.parent / "configs")
    p = argparse.ArgumentParser(description="Per-species mAP comparison across modes.")
    p.add_argument("--runs-dir", required=True, help="Training project dir (holds <mode>/weights/best.pt)")
    p.add_argument("--configs-dir", default=default_cfg, help="Directory holding dataset_*.yaml")
    p.add_argument("--out-dir", default=None, help="Where to write CSV + plots (default: --runs-dir)")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    df = evaluate(a.runs_dir, a.configs_dir)
    if df.empty:
        raise SystemExit("❌ No data generated. Check --runs-dir / --configs-dir.")

    out_dir = a.out_dir or a.runs_dir
    csv_path = os.path.join(out_dir, "class_comparison_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n\U0001f3c6 mAP50 by mode/species:")
    print(df.pivot(index="Mode", columns="Class", values="mAP50").to_string())
    print(f"\n\U0001f4c1 Saved {csv_path}")

    if not a.no_plots:
        plot_df = df.copy()
        plot_df["Mode"] = plot_df["Mode"].str.replace("_", " ").str.title()
        make_plot(plot_df, "mAP50", "Model accuracy by mode & species (mAP50)",
                  os.path.join(out_dir, "graph_mAP50_comparison.png"))
        make_plot(plot_df, "mAP50-95", "Model accuracy by mode & species (mAP50-95)",
                  os.path.join(out_dir, "graph_mAP50_95_comparison.png"))
