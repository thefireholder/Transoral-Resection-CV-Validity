#!/usr/bin/env python3
"""
SAM3, TEXT PROMPT (zero-shot condition of the SAM3 prompting ablation, Fig 3b).
For every test image and every concept in prompts.json, run SAM3 with the
concept's free-text prompt; each returned instance becomes a prediction:
  class = concepts[i]["class"][source_of_frame]   (skipped when null for that source)
  score = SAM3's instance score
  rle   = mask at original image resolution via seglib.rle_encode

Writes:
  data/predictions/sam3_text_test.json            uniform prediction file (158 images)
  data/computed/sam3_text_prompt_counts.csv        class, n_predictions, n_images_with_prediction

Self-contained: needs only transformers (Sam3Model/Sam3Processor), torch, numpy, cv2, PIL,
pycocotools -- NOT the sam2 package.

CLI defaults:
  --split test  --score-thresh 0.05  --device cuda  --prompts scripts/rerun/sam3/prompts.json
  --fp16 (recommended for 6 GB GPU)
  --max-images N  (debug: process only first N images)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
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
from seglib import COCO_1200, rle_encode, save_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--max-images", type=int, default=None, help="debug: stop after N images")
    args = ap.parse_args()

    # ---- load prompts -------------------------------------------------------
    prompts_data = json.loads(Path(args.prompts).read_text())
    source_map = prompts_data["source_of_frame"]   # prefix -> "JSR" | "JML"
    concepts   = prompts_data["concepts"]

    # ---- load model ---------------------------------------------------------
    from transformers import Sam3Model, Sam3Processor
    kw = {"torch_dtype": torch.float16} if args.fp16 else {}
    model     = Sam3Model.from_pretrained(args.model_id, **kw).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model_id)
    print(f"device {args.device}  fp16={args.fp16}  score_thresh={args.score_thresh}", flush=True)

    # ---- load COCO metadata -------------------------------------------------
    from pycocotools.coco import COCO
    coco    = COCO(str(COCO_1200 / "annotations" / f"instances_{args.split}.json"))
    img_dir = COCO_1200 / "images" / args.split
    all_imgs = coco.loadImgs(coco.getImgIds())
    if args.max_images:
        all_imgs = all_imgs[:args.max_images]
    print(f"processing {len(all_imgs)} images x {len(concepts)} concepts", flush=True)

    # ---- inference ----------------------------------------------------------
    per_image = {}
    counts = defaultdict(lambda: {"n_predictions": 0, "n_images_with_prediction": 0})

    for img_idx, info in enumerate(all_imgs):
        fn  = info["file_name"]
        H, W = info["height"], info["width"]
        # determine source vocabulary from filename prefix (part before "__")
        prefix = fn.split("__")[0]
        source = source_map.get(prefix)

        pil_img = Image.open(img_dir / fn).convert("RGB")
        entry   = per_image.setdefault(fn, {"file_name": fn, "height": H, "width": W, "predictions": []})
        img_had_pred = set()

        for concept in concepts:
            class_name = concept["class"].get(source) if source else None
            if class_name is None:
                continue   # this concept not annotated for this source

            prompt_text = concept["prompt"]
            inputs = processor(images=pil_img, text=prompt_text,
                               return_tensors="pt").to(args.device)
            if args.fp16:
                inputs = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                          for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Move to CPU before upsampling masks to avoid OOM on 6 GB GPU
            target_sizes = inputs["original_sizes"].tolist()
            outputs_cpu = type(outputs)(**{k: _cpu(v) for k, v in outputs.items()})
            torch.cuda.empty_cache()
            res = processor.post_process_instance_segmentation(
                outputs_cpu,
                threshold=args.score_thresh,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            for mask_t, score_t in zip(res["masks"], res["scores"]):
                score = float(score_t)
                m = np.asarray(mask_t.cpu(), dtype=np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if not m.any():
                    continue
                entry["predictions"].append({
                    "class": class_name,
                    "score": score,
                    "rle":   rle_encode((m > 0).astype(np.uint8)),
                })
                counts[class_name]["n_predictions"] += 1
                img_had_pred.add(class_name)

        for cls in img_had_pred:
            counts[cls]["n_images_with_prediction"] += 1

        if (img_idx + 1) % 10 == 0:
            total_preds = sum(len(e["predictions"]) for e in per_image.values())
            print(f"  {img_idx + 1}/{len(all_imgs)}  predictions so far: {total_preds}", flush=True)

    # ---- ensure every image has an entry (background frames) ----------------
    for info in coco.loadImgs(coco.getImgIds()):
        per_image.setdefault(info["file_name"], {
            "file_name": info["file_name"],
            "height":    info["height"],
            "width":     info["width"],
            "predictions": [],
        })

    # ---- write predictions --------------------------------------------------
    out = RESULT / "data" / "predictions" / f"sam3_text_{args.split}.json"
    save_predictions(out, "sam3_text", args.split, args.score_thresh, list(per_image.values()))
    print(f"wrote {out}  ({len(per_image)} images)", flush=True)

    # ---- write counts -------------------------------------------------------
    counts_path = RESULT / "data" / "computed" / "sam3_text_prompt_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(counts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "n_predictions", "n_images_with_prediction"])
        w.writeheader()
        for cls, row in sorted(counts.items()):
            w.writerow({"class": cls, **row})
    print(f"wrote {counts_path}")

    # ---- per-class summary --------------------------------------------------
    print("\nper-class prediction counts:")
    for cls, row in sorted(counts.items()):
        print(f"  {cls:40s}  preds={row['n_predictions']:4d}  images={row['n_images_with_prediction']:3d}")


if __name__ == "__main__":
    main()
