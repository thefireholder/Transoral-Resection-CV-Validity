#!/usr/bin/env python3
"""
Run the trained YOLO11n-seg checkpoint over a split (CPU is fine, ~3-5 min for
158 images) and save the masks in the uniform prediction format.

Weights: train_script/yolov12_project/runs/segment/1200images/1200images_train/weights/best.pt
Data   : processed_data/yolo_format/1200images  (class names from data.yaml)

Usage (from result/):
  conda activate pytorch_env
  python scripts/export_yolo_predictions.py               # test split, CPU
  python scripts/export_yolo_predictions.py --device 0    # GPU
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from seglib import SHARED, YOLO_1200, load_coco_gt, rle_encode, save_predictions

RESULT = Path(__file__).resolve().parents[1]
WEIGHTS = SHARED / "train_script/yolov12_project/runs/segment/1200images/1200images_train/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="keep every prediction above this score (same as Mask R-CNN export)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--out-name", default=None, help="file stem under data/predictions (default yolo_<split>)")
    args = ap.parse_args()

    torch.manual_seed(0)
    with open(YOLO_1200 / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    gt = load_coco_gt(args.split)              # only for the list of file names / sizes
    img_dir = YOLO_1200 / "images" / args.split
    files = sorted(img_dir.glob("*.jpg"))
    model = YOLO(str(args.weights))

    images = []
    for i in range(0, len(files), args.batch):
        chunk = [str(p) for p in files[i:i + args.batch]]
        results = model.predict(chunk, conf=args.conf, iou=0.7, imgsz=640, device=args.device,
                                retina_masks=True, verbose=False)
        for src_path, r in zip(chunk, results):   # results keep input order; r.path can be 'image0.jpg'
            fn = Path(src_path).name
            h, w = r.orig_shape
            preds = []
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy() > 0.5      # (N, H, W) at original res
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                for m, c, s in zip(masks, cls, conf):
                    if m.shape != (h, w):
                        import cv2
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if not m.any():
                        continue
                    preds.append({"class": names[int(c)], "score": float(s), "rle": rle_encode(m)})
            images.append({"file_name": fn, "height": int(h), "width": int(w), "predictions": preds})
        print(f"  {min(i + args.batch, len(files))}/{len(files)}", flush=True)

    out = RESULT / "data" / "predictions" / f"{args.out_name or f'yolo_{args.split}'}.json"
    save_predictions(out, args.out_name or "yolo", args.split, args.conf, images)
    print(f"wrote {out}  ({len(images)} images, {sum(len(i['predictions']) for i in images)} predictions)")


if __name__ == "__main__":
    main()
