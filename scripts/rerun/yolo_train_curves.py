#!/usr/bin/env python3
"""
Re-train YOLO11n-seg with the exact recipe of scripts/original/yolo/
yolo11n_model.py (60 epochs, imgsz 640, batch 16, seed 0, deterministic) but
with save_period=1, then evaluate EVERY epoch checkpoint on the train AND val
splits so the training curve has train-split precision / recall.

Writes data/training_curves/yolo_rerun.csv (collect_results.py prefers it
over the log-parsed yolo.csv) and data/predictions/yolo_rerun_test.json.

GPU. Training ~1 h (A100), per-epoch eval of 60 ckpts x 2 splits ~1-2 h.
Submit with:  sbatch scripts/rerun/yolo_train_curves.slurm

--eval-only <run_dir>  skips training and only evaluates existing epoch*.pt
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[1]
sys.path.insert(0, str(RESULT / "scripts"))
from seglib import YOLO_1200  # noqa: E402
DATA_YAML = str(YOLO_1200 / "data.yaml")
CURVE_COLS = ["epoch", "train_loss", "val_loss", "train_precision", "train_recall",
              "val_precision", "val_recall", "val_mAP50", "val_mAP50_95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval-only", type=Path, default=None, help="existing run dir with weights/epoch*.pt")
    ap.add_argument("--project", type=Path, default=HERE / "output" / "yolo")
    args = ap.parse_args()

    if args.eval_only:
        run_dir = args.eval_only
    else:
        model = YOLO("yolo11n-seg.pt")
        res = model.train(data=DATA_YAML, epochs=args.epochs, imgsz=640, batch=16, device=args.device,
                          workers=8, seed=0, deterministic=True, save_period=1,
                          project=str(args.project), name="1200images_train", exist_ok=True)
        run_dir = Path(res.save_dir)

    # losses per epoch from ultralytics' own results.csv
    loss = {}
    with open(run_dir / "results.csv") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v for k, v in r.items()}
            e = int(r["epoch"])
            loss[e] = (sum(float(r[f"train/{k}_loss"]) for k in ["box", "seg", "cls", "dfl"]),
                       sum(float(r[f"val/{k}_loss"]) for k in ["box", "seg", "cls", "dfl"]))

    ckpts = sorted(run_dir.glob("weights/epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    rows = []
    out_csv = RESULT / "data" / "training_curves" / "yolo_rerun.csv"
    for ck in ckpts:
        e = int(re.findall(r"\d+", ck.stem)[0]) + 1     # ultralytics: epoch0.pt = state after epoch 1 of results.csv
        m = YOLO(str(ck))
        r = {"epoch": e, "train_loss": loss.get(e, (None, None))[0], "val_loss": loss.get(e, (None, None))[1]}
        for split in ["train", "val"]:
            met = m.val(data=DATA_YAML, split=split, batch=32, device=args.device, plots=False, verbose=False,
                        project=str(args.project), name=f"epoch_eval_{split}", exist_ok=True)
            d = met.results_dict
            r[f"{split}_precision"] = d["metrics/precision(M)"]
            r[f"{split}_recall"] = d["metrics/recall(M)"]
            if split == "val":
                r["val_mAP50"] = d["metrics/mAP50(M)"]; r["val_mAP50_95"] = d["metrics/mAP50-95(M)"]
        rows.append(r)
        print(r, flush=True)
        with open(out_csv, "w", newline="") as f:      # rewrite every epoch so partial runs are usable
            w = csv.DictWriter(f, fieldnames=CURVE_COLS); w.writeheader()
            for rr in rows:
                w.writerow({k: ("n/p" if rr.get(k) is None else rr[k]) for k in CURVE_COLS})
    print("wrote", out_csv)

    # uniform test predictions from best.pt (same as scripts/export_yolo_predictions.py)
    import subprocess, sys
    subprocess.run([sys.executable, str(RESULT / "scripts" / "export_yolo_predictions.py"),
                    "--weights", str(run_dir / "weights" / "best.pt"), "--out-name", "yolo_rerun_test",
                    "--device", args.device], check=True)


if __name__ == "__main__":
    main()
