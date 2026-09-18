#!/usr/bin/env python3
"""
sam3_model.py — Evaluate SAM3 on the tonsil COCO dataset using GT bounding box prompts.

SAM3 is loaded via HuggingFace transformers (Sam3Model + Sam3Processor).
Metric utilities are imported from default_model.py.
Uses the same COCO dataset as default_model.py.
"""
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).parent))
from sam2_model import (
    EvalConfig,
    ap_per_class,
    coco_ann_to_binary_mask,
    coco_iou_thresholds,
    draw_box,
    load_image_rgb,
    mask_dice,
    mask_iou,
    overlay_mask,
    pad_box_xyxy,
    put_text,
    xywh_to_xyxy,
)


# -------------------------
# SAM3 Predictor
# -------------------------

class SAM3PredictorInterface:
    def predict_mask(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SAM3Predictor(SAM3PredictorInterface):
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str = "cuda",
        score_thresh: float = 0.5,
    ):
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.score_thresh = score_thresh
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    @torch.no_grad()
    def predict_mask(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """
        image_rgb: (H, W, 3) uint8 RGB
        box_xyxy:  (4,) float pixel coords [x1, y1, x2, y2]
        Returns binary mask (H, W) uint8 {0, 1}.
        """
        H, W = image_rgb.shape[:2]
        img_pil = Image.fromarray(image_rgb)

        inputs = self.processor(
            images=img_pil,
            input_boxes=[[box_xyxy.tolist()]],   # [batch=1, num_boxes=1, 4]
            input_boxes_labels=[[1]],              # 1 = positive prompt
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_thresh,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        if len(results["masks"]) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        scores = np.array([float(s) for s in results["scores"]])
        best = int(np.argmax(scores))
        m = np.asarray(results["masks"][best].cpu(), dtype=np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)


# -------------------------
# Evaluation
# -------------------------

def evaluate_sam3_on_coco_val(
    cfg: EvalConfig,
    predictor: SAM3PredictorInterface,
    result_path: Path,
) -> Dict[str, object]:

    ann_path = cfg.dataset_root / "annotations" / "instances_test.json"
    img_dir = cfg.dataset_root / "images" / "test"

    coco = COCO(str(ann_path))

    cats = coco.loadCats(coco.getCatIds())
    names = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    all_ious: List[float] = []
    all_dices: List[float] = []

    tp_rows: List[np.ndarray] = []
    target_cls: List[int] = []
    pred_cls: List[int] = []
    conf: List[float] = []

    ann_ids = coco.getAnnIds()
    if cfg.max_instances is not None:
        ann_ids = ann_ids[: cfg.max_instances]

    image_cache: Dict[int, Tuple[np.ndarray, dict]] = {}

    vis_dir = result_path / "outputs" / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_cache = {}

    for ann_id in ann_ids:
        ann = coco.loadAnns([ann_id])[0]
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if img_id not in image_cache:
            img_info = coco.loadImgs([img_id])[0]
            img_path = img_dir / img_info["file_name"]
            image_cache[img_id] = (load_image_rgb(img_path), img_info)

        image_rgb, img_info = image_cache[img_id]
        h, w = int(img_info["height"]), int(img_info["width"])

        gt_mask = coco_ann_to_binary_mask(coco, ann, h=h, w=w)
        if gt_mask.sum() == 0:
            continue

        box_xyxy = xywh_to_xyxy(np.array(ann["bbox"], dtype=float))
        box_xyxy = pad_box_xyxy(box_xyxy, cfg.pad_frac, img_w=w, img_h=h)

        pred_mask = predictor.predict_mask(image_rgb=image_rgb, box_xyxy=box_xyxy)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Pred mask shape {pred_mask.shape} != GT shape {gt_mask.shape} for ann_id={ann_id}"
            )

        iou = mask_iou(pred_mask, gt_mask)
        dice = mask_dice(pred_mask, gt_mask)
        all_ious.append(iou)
        all_dices.append(dice)

        tp_rows.append((iou >= cfg.iou_thrs).astype(np.uint8))
        target_cls.append(cat_id)
        pred_cls.append(cat_id)
        conf.append(1.0)

        # Visualization
        if img_id not in vis_cache:
            vis_cache[img_id] = image_rgb.copy()

        canvas = vis_cache[img_id]
        canvas = overlay_mask(canvas, gt_mask, (0, 255, 0), alpha=0.35)   # green = GT
        canvas = overlay_mask(canvas, pred_mask, (0, 0, 255), alpha=0.35) # red = pred
        canvas = draw_box(canvas, box_xyxy, (255, 0, 0))                  # blue = box
        label = f"{names.get(cat_id, cat_id)} | IoU {iou:.2f} | Dice {dice:.2f}"
        canvas = put_text(canvas, label, (int(box_xyxy[0]), max(15, int(box_xyxy[1]) - 5)))
        vis_cache[img_id] = canvas

    tp = np.stack(tp_rows, axis=0) if tp_rows else np.zeros((0, len(cfg.iou_thrs)), dtype=np.uint8)

    results = {
        "tp": tp,
        "conf": np.array(conf, dtype=float),
        "pred_cls": np.array(pred_cls, dtype=int),
        "target_cls": np.array(target_cls, dtype=int),
        "names_category_id_to_name": names,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "ious": np.array(all_ious, dtype=float),
        "dices": np.array(all_dices, dtype=float),
        "iou_thresholds": cfg.iou_thrs,
        "num_instances": int(len(all_ious)),
    }

    for img_id, canvas in vis_cache.items():
        img_info = coco.loadImgs([img_id])[0]
        save_path = vis_dir / img_info["file_name"]
        cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return results


# -------------------------
# Main
# -------------------------


def get_unique_path(path: Path) -> Path:
    base = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    new_path = path

    while new_path.exists():
        if suffix:
            new_name = f"{base}_{counter}{suffix}"
        else:
            new_name = f"{path.name}_{counter}"
        new_path = parent / new_name
        counter += 1

    return new_path

if __name__ == "__main__":
    # dataset_root = Path("/u/slee32/data/vinay_ent/JS_R_Tonsil_coco_merged_v3_vv4")
    dataset_root = Path("/projects/illinois/cimed/bts/gayed/data/processed_data/coco_format/1200images")
    result_path = Path("/projects/illinois/cimed/bts/gayed/data/train_script/SAM2_project/sam2/result/sam3/"+dataset_root.stem)
    result_path = get_unique_path(result_path)

    cfg = EvalConfig(dataset_root=dataset_root, pad_frac=0.05, max_instances=None)

    predictor = SAM3Predictor(
        model_id="facebook/sam3",
        device="cuda",
        score_thresh=0.5,
    )

    out = evaluate_sam3_on_coco_val(cfg, predictor, result_path)

    print("Instances:", out["num_instances"])
    print("Mean IoU:", out["mean_iou"])
    print("Mean Dice:", out["mean_dice"])

    tp_, fp_, p, r, f1, ap, unique_classes, *_ = ap_per_class(
        tp=out["tp"],
        conf=out["conf"],
        pred_cls=out["pred_cls"],
        target_cls=out["target_cls"],
        plot=True,
        names=out["names_category_id_to_name"],
        save_dir=result_path,
    )

    names = out["names_category_id_to_name"]

    print("\nPer-class AP:")
    print("-" * 60)
    for i, cls_id in enumerate(unique_classes):
        cls_name = names.get(cls_id, str(cls_id))
        print(f"{cls_name:20s} | AP@0.5: {ap[i, 0]:.4f} | AP@0.5:0.95: {ap[i].mean():.4f}")

    per_class_ap = {
        names.get(cls_id, str(cls_id)): {
            "AP@0.5": float(ap[i, 0]),
            "AP@0.5:0.95": float(ap[i].mean()),
        }
        for i, cls_id in enumerate(unique_classes)
    }
    result_path.mkdir(exist_ok=True)
    with open(result_path / "per_class_ap.json", "w") as f:
        json.dump(per_class_ap, f, indent=2)

    iou_by_class: Dict[int, list] = defaultdict(list)
    dice_by_class: Dict[int, list] = defaultdict(list)
    for i, cls_id in enumerate(out["target_cls"]):
        iou_by_class[cls_id].append(out["ious"][i])
        dice_by_class[cls_id].append(out["dices"][i])

    print("\nPer-class IoU / Dice:")
    print("-" * 60)
    for cls_id, ious in iou_by_class.items():
        cls_name = names.get(cls_id, str(cls_id))
        print(
            f"{cls_name:20s} | "
            f"Mean IoU: {np.mean(ious):.4f} | "
            f"Mean Dice: {np.mean(dice_by_class[cls_id]):.4f}"
        )

    print("mAP@0.5:", ap[:, 0].mean())
    print("mAP@0.5:0.95:", ap.mean())
