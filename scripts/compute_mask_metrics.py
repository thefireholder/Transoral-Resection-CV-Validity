#!/usr/bin/env python3
"""
Score a uniform prediction file against the COCO ground truth with the shared
matching code in seglib.py and write:

  data/pr_curves/<model>.json            per-class mask PR curve @IoU0.5 (Figure 2)
  data/computed/<model>_<split>.json     per-class + aggregate AP50 / AP50:95 /
                                         precision / recall / IoU / Dice

The numbers in data/computed/ are *recomputed here with one common
implementation* (COCO 101-point AP by default, --ap-method ultralytics for
the ultralytics rule); the ones in data/metrics_*.csv tagged source=reported
come from each model's own original evaluation and are collected by
collect_results.py. With --ap-method coco, Mask R-CNN agrees with its
COCOeval numbers to ~0.01; with --ap-method ultralytics, YOLO agrees with
its ultralytics numbers to ~0.01.

CPU only. ~1 min per model.

Usage (from result/):
  conda activate pytorch_env
  python scripts/compute_mask_metrics.py maskrcnn
  python scripts/compute_mask_metrics.py yolo
  python scripts/compute_mask_metrics.py sam3          # after scripts/rerun/sam3_eval_save_predictions.py
  python scripts/compute_mask_metrics.py --all
"""
import argparse
import json
from pathlib import Path

from seglib import evaluate_predictions, load_coco_gt, load_predictions, save_pr_curves

RESULT = Path(__file__).resolve().parents[1]
MODELS = ["maskrcnn", "yolo", "sam3", "monai"]


def run(model: str, split: str, ap_method: str):
    pred_path = RESULT / "data" / "predictions" / f"{model}_{split}.json"
    if not pred_path.exists():
        print(f"[skip] {pred_path} not found (n/p)")
        return
    gt = load_coco_gt(split)
    res = evaluate_predictions(load_predictions(pred_path), gt, ap_method=ap_method)
    save_pr_curves(RESULT / "data" / "pr_curves" / f"{model}.json", model, res)
    out = RESULT / "data" / "computed" / f"{model}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": model, "split": split, "source": str(pred_path),
                   "ap_method": ap_method,
                   "aggregate": res["aggregate"], "per_class": res["per_class"]}, f, indent=2)
    a = res["aggregate"]
    print(f"{model:9s} {split}: mAP50={a['mAP50']:.4f} mAP50-95={a['mAP50_95']:.4f} "
          f"IoU={a['iou']:.4f} Dice={a['dice']:.4f}  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ap-method", default="coco", choices=["coco", "ultralytics"],
                    help="AP integration rule; see seglib._compute_ap (default coco)")
    args = ap.parse_args()
    for m in (MODELS if args.all else args.models):
        run(m, args.split, args.ap_method)


if __name__ == "__main__":
    main()
