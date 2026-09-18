#!/usr/bin/env python3
"""
Train + test YOLO11n-seg on one training subset (same recipe as the 1200images run).
  python yolo_sweep.py --n 150 [--seed 0] [--epochs 60]
Appends one row to data/dataset_size_sweep.csv (via collect_sweep.py).
"""
import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    sub = HERE / "subsets" / "yolo" / f"n{args.n}"
    run_dir = HERE / "runs" / "yolo" / f"n{args.n}_seed{args.seed}"
    model = YOLO("yolo11n-seg.pt")
    model.train(data=str(sub / "data.yaml"), epochs=args.epochs, imgsz=640, batch=16, device=args.device,
                workers=8, seed=args.seed, deterministic=True, project=str(run_dir), name="train", exist_ok=True)
    met = model.val(data=str(sub / "data.yaml"), split="test", batch=32, device=args.device,
                    project=str(run_dir), name="test", exist_ok=True)
    d = met.results_dict
    row = {"model": "yolo", "split": "test", "seed": args.seed, "n_train": args.n,
           "mAP50": d["metrics/mAP50(M)"], "mAP50_95": d["metrics/mAP50-95(M)"],
           "precision": d["metrics/precision(M)"], "recall": d["metrics/recall(M)"],
           "iou": None, "dice": None, "run_dir": str(run_dir)}
    json.dump(row, open(run_dir / "sweep_row.json", "w"), indent=1)
    print(row)
    sys.path.insert(0, str(HERE)); import collect_sweep; collect_sweep.main()


if __name__ == "__main__":
    main()
