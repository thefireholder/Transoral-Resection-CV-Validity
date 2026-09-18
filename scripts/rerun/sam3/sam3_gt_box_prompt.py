#!/usr/bin/env python3
"""
SAM3, GT-BOX PROMPT ("oracle" condition of the SAM3 prompting ablation, Fig 3b).
Re-runs the evaluation of scripts/original/sam/sam3_model.py (GT bounding box
prompt, padded 5%, one mask per GT instance, test split) and saves what the
original run threw away:

  data/predictions/sam3_gtbox_test.json      uniform prediction file (masks + SAM3 mask score)
                                             -> scripts/compute_mask_metrics.py sam3_gtbox
  data/computed/sam3_gtbox_instances_test.csv   per-instance IoU / Dice / score

Sibling scripts in this folder: sam3_text_prompt.py (zero-shot, the SAM3 used in
the main figures) and sam3_yolo_box_prompt.py (two-stage YOLO boxes -> SAM3).

Self-contained: needs only transformers (Sam3Model/Sam3Processor), torch,
pycocotools, numpy, PIL, cv2 -- NOT the sam2 package. The helper functions
are copied verbatim from scripts/original/sam/sam2_model.py.

Differences from the original run (documented, on purpose):
  * "score" = SAM3's own score for the chosen mask (the original assigned
    conf = 1.0 to everything, which makes any PR curve a single point).
    AP50 / Dice / IoU are unaffected except through ranking.
  * seed fixed (torch / numpy / random = 0). SAM3 inference is deterministic
    anyway, so numbers should match logs/sam3_eval_9794464.out closely.

GPU, ~1 h on A100; ~2-3 h on a 6 GB laptop GPU (model is ~3.5 GB in fp32;
use --fp16 if it does not fit).
Weights: facebook/sam3 (gated on HuggingFace). On the cluster they are cached
in HF_HOME=/u/sl257/scratch/huggingface; elsewhere log in once with
`huggingface-cli login` and let it download.
Submit with:  sbatch scripts/rerun/sam3/sam3_gt_box_prompt.slurm
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "scripts"))                       # seglib
from seglib import COCO_1200, coco_ann_to_rle, rle_decode, rle_encode, save_predictions  # noqa: E402


# ---- copied from scripts/original/sam/sam2_model.py -----------------------------------------
def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh.astype(float)
    return np.array([x, y, x + w, y + h], dtype=float)


def pad_box_xyxy(box_xyxy, pad_frac, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad_frac, bh * pad_frac
    return np.array([max(0.0, x1 - px), max(0.0, y1 - py),
                     min(float(img_w - 1), x2 + px), min(float(img_h - 1), y2 + py)], dtype=float)


def mask_iou(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return float(np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + eps))


def mask_dice(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + eps))
# ---------------------------------------------------------------------------------------------


class SAM3BoxPredictor:
    """Same as sam3_model.SAM3Predictor but also returns the chosen mask's score."""

    def __init__(self, model_id="facebook/sam3", device="cuda", score_thresh=0.5, fp16=False):
        from transformers import Sam3Model, Sam3Processor
        self.device, self.score_thresh = device, score_thresh
        kw = {"torch_dtype": torch.float16} if fp16 else {}
        self.model = Sam3Model.from_pretrained(model_id, **kw).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.fp16 = fp16

    @torch.no_grad()
    def __call__(self, image_rgb, box_xyxy):
        H, W = image_rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(image_rgb), input_boxes=[[box_xyxy.tolist()]],
                                input_boxes_labels=[[1]], return_tensors="pt").to(self.device)
        if self.fp16:
            inputs = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.score_thresh, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist())[0]
        if len(res["masks"]) == 0:
            return np.zeros((H, W), dtype=np.uint8), 0.0
        scores = np.array([float(s) for s in res["scores"]])
        best = int(np.argmax(scores))
        m = np.asarray(res["masks"][best].cpu(), dtype=np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8), float(scores[best])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pad-frac", type=float, default=0.05)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", help="half precision (for small GPUs)")
    ap.add_argument("--max-instances", type=int, default=None, help="debug: stop early")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from pycocotools.coco import COCO
    coco = COCO(str(COCO_1200 / "annotations" / f"instances_{args.split}.json"))
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_dir = COCO_1200 / "images" / args.split
    predictor = SAM3BoxPredictor(args.model_id, args.device, args.score_thresh, args.fp16)

    per_image, inst_rows, cache = {}, [], {}
    ann_ids = coco.getAnnIds()[: args.max_instances]
    for k, ann_id in enumerate(ann_ids):
        ann = coco.loadAnns([ann_id])[0]
        info = coco.loadImgs([ann["image_id"]])[0]
        fn = info["file_name"]
        if fn not in cache:
            cache = {fn: np.array(Image.open(img_dir / fn).convert("RGB"))}   # keep one image in memory
        img = cache[fn]
        h, w = info["height"], info["width"]
        gt_rle = coco_ann_to_rle(ann, h, w)
        if gt_rle is None:
            continue
        box = pad_box_xyxy(xywh_to_xyxy(np.array(ann["bbox"], dtype=float)), args.pad_frac, w, h)
        pred, score = predictor(img, box)
        gt = rle_decode(gt_rle)
        iou, dice = mask_iou(pred, gt), mask_dice(pred, gt)
        entry = per_image.setdefault(fn, {"file_name": fn, "height": h, "width": w, "predictions": []})
        if pred.any():
            entry["predictions"].append({"class": names[ann["category_id"]], "score": score, "rle": rle_encode(pred)})
        inst_rows.append({"ann_id": ann_id, "file_name": fn, "class": names[ann["category_id"]],
                          "score": score, "iou": iou, "dice": dice})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(ann_ids)}  running mean IoU {np.mean([r['iou'] for r in inst_rows]):.4f}", flush=True)

    for info in coco.loadImgs(coco.getImgIds()):          # background frames need an (empty) entry too
        per_image.setdefault(info["file_name"], {"file_name": info["file_name"], "height": info["height"],
                                                 "width": info["width"], "predictions": []})
    out = RESULT / "data" / "predictions" / f"sam3_gtbox_{args.split}.json"
    save_predictions(out, "sam3_gtbox", args.split, 0.0, list(per_image.values()))
    inst_path = RESULT / "data" / "computed" / f"sam3_gtbox_instances_{args.split}.csv"
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inst_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst_rows[0])); w.writeheader(); w.writerows(inst_rows)
    print(f"instances {len(inst_rows)}  mean IoU {np.mean([r['iou'] for r in inst_rows]):.4f}  "
          f"mean Dice {np.mean([r['dice'] for r in inst_rows]):.4f}")
    print("wrote", out, "and", inst_path)


if __name__ == "__main__":
    main()
