#!/usr/bin/env python3
"""
SAM3, YOLO-BOX PROMPT (two-stage condition of the SAM3 prompting ablation, Fig 3b).
Reads data/predictions/yolo_test.json (the original cluster YOLO model), converts
each YOLO prediction's RLE mask to a bounding box, prompts SAM3 with that box (padded
5%), and replaces the mask with SAM3's best-scoring output. Class and score remain
YOLO's so the two rows are directly comparable.

  For each YOLO prediction (score >= 0.05):
    box = toBbox(rle) -> xywh -> xyxy -> pad 5%
    SAM3(box) -> highest-scoring mask at original resolution
    if SAM3 returns no mask: keep YOLO mask unchanged, increment fallback counter
  class = YOLO class, score = YOLO score, rle = SAM3 mask (or original YOLO mask)

Writes:
  data/predictions/sam3_yolobox_test.json    model="sam3_yolobox", 158 images

Self-contained: needs only transformers, torch, numpy, cv2, PIL, pycocotools -- NOT sam2.

CLI defaults:
  --split test  --score-thresh 0.05  --device cuda  --fp16
  --yolo-preds data/predictions/yolo_test.json
  --max-images N  (debug)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(i) for i in x)
    return x

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "scripts"))
from seglib import COCO_1200, rle_decode, rle_encode, save_predictions  # noqa: E402


# ---- copied from sam3_gt_box_prompt.py --------------------------------------
def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh.astype(float)
    return np.array([x, y, x + w, y + h], dtype=float)


def pad_box_xyxy(box_xyxy, pad_frac, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad_frac, bh * pad_frac
    return np.array([max(0.0, x1 - px), max(0.0, y1 - py),
                     min(float(img_w - 1), x2 + px), min(float(img_h - 1), y2 + py)], dtype=float)
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thresh", type=float, default=0.05,
                    help="minimum YOLO score to prompt SAM3 (also written as file score_threshold)")
    ap.add_argument("--pad-frac", type=float, default=0.05)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--yolo-preds", default=str(RESULT / "data" / "predictions" / "yolo_test.json"))
    ap.add_argument("--max-images", type=int, default=None, help="debug: stop after N images")
    args = ap.parse_args()

    # ---- load YOLO predictions ----------------------------------------------
    yolo_data = json.loads(Path(args.yolo_preds).read_text())
    yolo_by_fn = {img["file_name"]: img for img in yolo_data["images"]}
    print(f"loaded {len(yolo_by_fn)} images from {args.yolo_preds}", flush=True)

    total_boxes = sum(
        len([p for p in img["predictions"] if p["score"] >= args.score_thresh])
        for img in yolo_data["images"]
    )
    print(f"boxes to prompt (score >= {args.score_thresh}): {total_boxes}", flush=True)

    # ---- load model ---------------------------------------------------------
    from transformers import Sam3Model, Sam3Processor
    kw = {"torch_dtype": torch.float16} if args.fp16 else {}
    model     = Sam3Model.from_pretrained(args.model_id, **kw).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model_id)
    print(f"device {args.device}  fp16={args.fp16}", flush=True)

    # ---- load image list from COCO (to ensure all 158 images are covered) ---
    from pycocotools.coco import COCO
    import pycocotools.mask as mask_util
    coco    = COCO(str(COCO_1200 / "annotations" / f"instances_{args.split}.json"))
    img_dir = COCO_1200 / "images" / args.split
    all_imgs = coco.loadImgs(coco.getImgIds())
    if args.max_images:
        all_imgs = all_imgs[:args.max_images]

    # ---- inference ----------------------------------------------------------
    per_image  = {}
    n_fallback = 0   # boxes where SAM3 returned nothing -> kept YOLO mask
    n_prompted = 0

    for img_idx, info in enumerate(all_imgs):
        fn  = info["file_name"]
        H, W = info["height"], info["width"]
        entry = {"file_name": fn, "height": H, "width": W, "predictions": []}

        yolo_img = yolo_by_fn.get(fn)
        if yolo_img is None or not yolo_img["predictions"]:
            per_image[fn] = entry
            continue

        pil_img = Image.open(img_dir / fn).convert("RGB")

        for pred in yolo_img["predictions"]:
            if pred["score"] < args.score_thresh:
                continue

            # box from YOLO RLE mask
            xywh = np.array(mask_util.toBbox([pred["rle"]])[0], dtype=float)
            box  = pad_box_xyxy(xywh_to_xyxy(xywh), args.pad_frac, W, H)

            # SAM3 box prompt
            inputs = processor(images=pil_img, input_boxes=[[box.tolist()]],
                               input_boxes_labels=[[1]], return_tensors="pt").to(args.device)
            if args.fp16:
                inputs = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                          for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Move to CPU before upsampling masks to avoid OOM on 6 GB GPU
            target_sizes = inputs["original_sizes"].tolist()
            outputs_cpu = type(outputs)(**{k: _cpu(v) for k, v in outputs.items()})
            torch.cuda.empty_cache()
            # use score_thresh=0.0 here to always get the best mask back
            res = processor.post_process_instance_segmentation(
                outputs_cpu, threshold=0.0, mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            n_prompted += 1
            if len(res["masks"]) == 0:
                # SAM3 returned nothing: keep original YOLO mask
                n_fallback += 1
                entry["predictions"].append({
                    "class": pred["class"],
                    "score": pred["score"],
                    "rle":   pred["rle"],
                })
            else:
                scores = np.array([float(s) for s in res["scores"]])
                best   = int(np.argmax(scores))
                m = np.asarray(res["masks"][best].cpu(), dtype=np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if not m.any():
                    n_fallback += 1
                    entry["predictions"].append({
                        "class": pred["class"],
                        "score": pred["score"],
                        "rle":   pred["rle"],
                    })
                else:
                    entry["predictions"].append({
                        "class": pred["class"],
                        "score": pred["score"],   # YOLO's score, not SAM3's
                        "rle":   rle_encode((m > 0).astype(np.uint8)),
                    })

        per_image[fn] = entry

        if (img_idx + 1) % 10 == 0:
            print(f"  {img_idx + 1}/{len(all_imgs)}  prompted {n_prompted}  fallback {n_fallback}", flush=True)

    # ---- ensure all 158 images present (background frames) ------------------
    for info in coco.loadImgs(coco.getImgIds()):
        per_image.setdefault(info["file_name"], {
            "file_name": info["file_name"],
            "height":    info["height"],
            "width":     info["width"],
            "predictions": [],
        })

    print(f"\ntotal boxes prompted: {n_prompted}  SAM3 fallback (kept YOLO mask): {n_fallback}", flush=True)

    out = RESULT / "data" / "predictions" / f"sam3_yolobox_{args.split}.json"
    save_predictions(out, "sam3_yolobox", args.split, args.score_thresh, list(per_image.values()))
    print(f"wrote {out}  ({len(per_image)} images)")


if __name__ == "__main__":
    main()
