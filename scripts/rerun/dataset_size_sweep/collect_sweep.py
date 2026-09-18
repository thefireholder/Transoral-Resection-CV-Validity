#!/usr/bin/env python3
"""
Gather every runs/*/*/sweep_row.json into data/dataset_size_sweep.csv
(rewrites the file; one row per model x n_train x seed). Run any time.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
COLS = ["model", "split", "seed", "n_train", "mAP50", "mAP50_95", "precision", "recall", "iou", "dice", "run_dir"]


def main():
    rows = [json.load(open(p)) for p in sorted((HERE / "runs").glob("*/*/sweep_row.json"))]
    out = RESULT / "data" / "dataset_size_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["n_train"], r["seed"])):
            w.writerow({k: ("n/p" if r.get(k) is None else r[k]) for k in COLS})
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
