import os
from ultralytics import YOLO

# Mode -> experiment folder name (weights live at <runs-dir>/<exp_name>/weights/best.pt)
MODE_EXP = {
    0: "0_baseline",
    1: "1_tiling_naive",
    2: "2_tiling_overlapping",
    3: "3_smart_roi",
    4: "4_baseline_corrected",
    5: "5_tiling_naive_corrected",
    6: "6_tiling_overlapping_corrected",
    7: "7_smart_roi_corrected",
}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run inference/validation for a trained mode.")
    p.add_argument("--mode", type=int, required=True, choices=sorted(MODE_EXP))
    p.add_argument("--runs-dir", required=True, help="Training project dir (holds <exp>/weights/best.pt)")
    p.add_argument("--test-dir", required=True, help="Folder of images to run prediction on")
    p.add_argument("--predict-dir", required=True, help="Where annotated predictions are saved")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()

    exp_name = MODE_EXP[a.mode]
    model_path = f"{a.runs_dir}/{exp_name}/weights/best.pt"
    print(f"\U0001f680 Inference for mode {a.mode} using {model_path}")

    try:
        model = YOLO(model_path)
    except Exception:
        print(f"\u274c Could not load model at {model_path} -- train this mode first.")
        raise SystemExit(1)

    model.predict(
        source=a.test_dir,
        save=True,
        show=False,
        conf=a.conf,
        project=a.predict_dir,
        name=exp_name,
    )
    print(f"\u2705 Predictions saved to: {a.predict_dir}/{exp_name}")
