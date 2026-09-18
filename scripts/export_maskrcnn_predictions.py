#!/usr/bin/env python3
"""
Convert the Mask R-CNN predictions already saved by
  train_script/maskrcnn_project/eval_saved_mask_metrics.py
  -> run_output/mask_metrics/mask_metrics_{split}.json
into the uniform prediction format (see seglib.py docstring).

CPU only, no GPU, no model needed. ~10 s.

Usage (from result/):
  conda activate pytorch_env
  python scripts/export_maskrcnn_predictions.py            # test split
  python scripts/export_maskrcnn_predictions.py --split val
"""
import argparse
import json
from pathlib import Path

from seglib import SHARED, save_predictions

RESULT = Path(__file__).resolve().parents[1]
MRCNN_RUN = SHARED / "train_script/maskrcnn_project/run_output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    src = MRCNN_RUN / "mask_metrics" / f"mask_metrics_{args.split}.json"
    ann = MRCNN_RUN / "mask_metrics" / f"instances_{args.split}.filtered_for_mask_metrics.json"
    with open(src) as f:
        d = json.load(f)
    with open(ann) as f:
        coco = json.load(f)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    im_info = {im["id"]: im for im in coco["images"]}

    images = []
    for entry in d["predictions"]:
        im = im_info[entry["image_id"]]
        preds = [{"class": cat_name[p["label"]], "score": p["score"], "rle": p["segmentation"]}
                 for p in entry["predictions"]]
        images.append({"file_name": im["file_name"], "height": im["height"],
                       "width": im["width"], "predictions": preds})

    out = RESULT / "data" / "predictions" / f"maskrcnn_{args.split}.json"
    save_predictions(out, "maskrcnn", args.split, d["score_threshold"], images)
    print(f"wrote {out}  ({len(images)} images, "
          f"{sum(len(i['predictions']) for i in images)} predictions, source={src})")


if __name__ == "__main__":
    main()
