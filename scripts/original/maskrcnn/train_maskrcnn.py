#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


class SurgicalMaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, image_size: int = 1024):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.ann_path = root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(self.ann_path))
        self.image_ids = list(self.coco.imgs.keys())
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.img_dir = root / "images" / split

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.img_dir / img_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            x2 = max(0, min(width, x + w))
            y2 = max(0, min(height, y + h))
            x = max(0, min(width, x))
            y = max(0, min(height, y))
            if x2 <= x or y2 <= y:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
                if len(coords) < 3:
                    continue
                coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
                coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
            if mask.sum() == 0:
                continue
            boxes.append([x, y, x2, y2])
            labels.append(self.cat_id_to_idx[ann["category_id"]])
            masks.append(mask)

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, height, width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)

        image_tensor = F.to_tensor(image)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def is_valid_polygon_segmentation(segmentation):
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return False
    for poly in segmentation:
        if not isinstance(poly, list):
            return False
        if len(poly) < 6 or len(poly) % 2 != 0:
            return False
    return True


def build_filtered_coco_gt(src_ann_path: Path, output_dir: Path, output_name: str | None = None) -> COCO:
    with open(src_ann_path) as f:
        coco_json = json.load(f)

    filtered_annotations = [
        ann
        for ann in coco_json.get("annotations", [])
        if is_valid_polygon_segmentation(ann.get("segmentation"))
    ]
    dropped = len(coco_json.get("annotations", [])) - len(filtered_annotations)
    if dropped > 0:
        print(f"Dropped {dropped} invalid GT polygon annotation(s) before COCO segm eval")

    coco_json["annotations"] = filtered_annotations
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / (output_name or "instances_val.filtered_for_eval.json")
    with open(filtered_path, "w") as f:
        json.dump(coco_json, f)
    return COCO(str(filtered_path))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes: int, device: torch.device, weights="DEFAULT"):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)
    model.to(device)
    return model


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    count = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in t.items()
        } for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        count += 1
    return total_loss / max(1, count)


def evaluate(model, loader, coco_gt, device, score_thresh=0.05, cat_id_to_idx=None):
    model.eval()
    results = []
    idx_to_cat_id = {v: k for k, v in (cat_id_to_idx or {}).items()}
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                for box, label, score, mask in zip(
                    output["boxes"], output["labels"], output["scores"], output["masks"]
                ):
                    if score < score_thresh:
                        continue
                    box = box.detach().cpu().numpy().astype(float)
                    label_idx = int(label.item())
                    category_id = idx_to_cat_id.get(label_idx, label_idx)
                    mask = mask[0].detach().cpu().numpy().astype(np.float32)
                    mask = np.squeeze(mask)
                    mask = (mask > 0.5).astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    counts = rle.get("counts")
                    if isinstance(counts, (bytes, bytearray)):
                        rle["counts"] = counts.decode("utf-8")
                    elif isinstance(counts, np.ndarray):
                        rle["counts"] = counts.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score.item()),
                        "segmentation": rle,
                    })
    if len(results) == 0:
        print("No predictions passed the score threshold; skipping COCO evaluation")
        return {
            "bbox_stats": None,
            "segm_stats": None,
            "num_results": 0,
        }

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval_segm = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()

    return {
        "bbox_stats": [float(x) for x in coco_eval.stats.tolist()],
        "segm_stats": [float(x) for x in coco_eval_segm.stats.tolist()],
        "num_results": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/projects/illinois/cimed/bts/gayed/data/processed_data/coco_format/maskrcnn"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/run_output"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SurgicalMaskRCNNDataset(args.root, split="train")
    val_ds = SurgicalMaskRCNNDataset(args.root, split="val")
    test_ds = SurgicalMaskRCNNDataset(args.root, split="test")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    num_classes = len(train_ds.cat_ids) + 1
    model = get_model(num_classes, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    coco_gt = build_filtered_coco_gt(
        args.root / "annotations" / "instances_val.json",
        args.output_dir,
        output_name="instances_val.filtered_for_eval.json",
    )

    best_val_ap = -1.0
    best_epoch = -1
    best_ckpt_path = args.output_dir / "maskrcnn_resnet50_fpn_best.pth"

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.4f}")
        val_stats = evaluate(model, val_loader, coco_gt, device, cat_id_to_idx=val_ds.cat_id_to_idx)
        scheduler.step()

        if val_stats and val_stats["bbox_stats"] is not None:
            val_ap = val_stats["bbox_stats"][0]  # AP@[0.50:0.95]
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  -> New best val bbox AP: {best_val_ap:.4f} (epoch {best_epoch}), saved to {best_ckpt_path}")

    print(f"\nBest val bbox AP: {best_val_ap:.4f} at epoch {best_epoch}")
    print(f"Loading best checkpoint from {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_coco_gt = build_filtered_coco_gt(
        args.root / "annotations" / "instances_test.json",
        args.output_dir,
        output_name="instances_test.filtered_for_eval.json",
    )
    print("Evaluating on test set")
    test_stats = evaluate(model, test_loader, test_coco_gt, device, cat_id_to_idx=test_ds.cat_id_to_idx)

    # Save metrics JSON
    metrics = {
        "best_epoch": best_epoch,
        "best_val_bbox_ap": best_val_ap,
        "val": val_stats,
        "test": test_stats,
    }
    metrics_path = args.output_dir / "maskrcnn_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON to {metrics_path}")
    print(f"Saved best model to {best_ckpt_path}")


if __name__ == "__main__":
    main()
