#!/usr/bin/env python3
"""Evaluate a saved Mask R-CNN checkpoint with per-class mask metrics."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path("/projects/illinois/cimed/bts/gayed")
sys.path.append(str(ROOT / "data/train_script/maskrcnn_project"))
from train_maskrcnn import (
    SurgicalMaskRCNNDataset,
    build_filtered_coco_gt,
    collate_fn,
    get_model,
    set_seed,
)


def mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union else 0.0


def mask_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denominator = pred_mask.sum() + gt_mask.sum()
    return float(2.0 * intersection / denominator) if denominator else 0.0


def rle_for_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": rle["size"], "counts": counts}


def evaluate_split(split, model, device, root, output_dir, score_threshold, model_path):
    dataset = SurgicalMaskRCNNDataset(root, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    annotation_path = root / "annotations" / f"instances_{split}.json"
    coco_gt = build_filtered_coco_gt(
        annotation_path,
        output_dir,
        output_name=f"instances_{split}.filtered_for_mask_metrics.json",
    )
    idx_to_cat_id = {value: key for key, value in dataset.cat_id_to_idx.items()}
    category_names = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

    coco_results = []
    matched_ious = defaultdict(list)
    matched_dices = defaultdict(list)
    saved_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images_gpu = [image.to(device) for image in images]
            outputs = model(images_gpu)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                gt_masks = target["masks"].numpy().astype(bool)
                gt_labels = target["labels"].numpy().astype(int)
                gt_by_label = defaultdict(list)
                for gt_mask, label in zip(gt_masks, gt_labels):
                    gt_by_label[int(label)].append(gt_mask)

                predictions_by_label = defaultdict(list)
                image_results = {
                    "image_id": image_id,
                    "predictions": [],
                }
                for box, label, score, mask in zip(
                    output["boxes"], output["labels"], output["scores"], output["masks"]
                ):
                    score_value = float(score.item())
                    if score_value < score_threshold:
                        continue
                    label_idx = int(label.item())
                    category_id = idx_to_cat_id.get(label_idx, label_idx)
                    binary_mask = (mask[0].cpu().numpy() > 0.5)
                    box_values = box.cpu().numpy().astype(float)
                    prediction = {
                        "score": score_value,
                        "label": category_id,
                        "bbox": [
                            float(box_values[0]),
                            float(box_values[1]),
                            float(box_values[2] - box_values[0]),
                            float(box_values[3] - box_values[1]),
                        ],
                        "segmentation": rle_for_mask(binary_mask),
                    }
                    image_results["predictions"].append(prediction)
                    predictions_by_label[label_idx].append((score_value, binary_mask))
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": prediction["bbox"],
                        "score": score_value,
                        "segmentation": prediction["segmentation"],
                    })

                for label_idx, predictions in predictions_by_label.items():
                    predictions.sort(key=lambda item: item[0], reverse=True)
                    used_gt = [False] * len(gt_by_label[label_idx])
                    for _, predicted_mask in predictions:
                        best_index = None
                        best_iou = 0.0
                        for gt_index, ground_truth_mask in enumerate(gt_by_label[label_idx]):
                            if used_gt[gt_index]:
                                continue
                            current_iou = mask_iou(predicted_mask, ground_truth_mask)
                            if current_iou > best_iou:
                                best_iou = current_iou
                                best_index = gt_index
                        if best_index is not None and best_iou >= 0.5:
                            used_gt[best_index] = True
                            ground_truth_mask = gt_by_label[label_idx][best_index]
                            category_id = idx_to_cat_id[label_idx]
                            matched_ious[category_id].append(best_iou)
                            matched_dices[category_id].append(
                                mask_dice(predicted_mask, ground_truth_mask)
                            )

                saved_predictions.append(image_results)

            if (batch_idx + 1) % 10 == 0:
                print(f"  {split}: {(batch_idx + 1) * len(images)}/{len(dataset)} images", flush=True)

    if not coco_results:
        raise RuntimeError(f"No predictions passed score threshold for {split}")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    category_metrics = {}
    category_ids = coco_gt.getCatIds()
    for category_index, category_id in enumerate(category_ids):
        precision = coco_eval.eval["precision"][:, :, category_index, 0, -1]
        valid_precision = precision[precision > -1]
        ap_all = float(np.mean(valid_precision)) if valid_precision.size else None
        ap_50_values = coco_eval.eval["precision"][0, :, category_index, 0, -1]
        valid_ap_50 = ap_50_values[ap_50_values > -1]
        ap_50 = float(np.mean(valid_ap_50)) if valid_ap_50.size else None
        ious = matched_ious.get(category_id, [])
        dices = matched_dices.get(category_id, [])
        category_metrics[category_names[category_id]] = {
            "category_id": category_id,
            "mask_ap_0.5": ap_50,
            "mask_ap_0.5_0.95": ap_all,
            "mean_mask_iou": float(np.mean(ious)) if ious else None,
            "mean_mask_dice": float(np.mean(dices)) if dices else None,
            "matched_instances": len(ious),
        }

    def aggregate(metric_by_category):
        values = [value for value in metric_by_category if value is not None]
        return float(np.mean(values)) if values else None

    aggregate_metrics = {
        "mask_mAP_0.5": float(coco_eval.stats[1]),
        "mask_mAP_0.5_0.95": float(coco_eval.stats[0]),
        "mean_mask_iou": aggregate([value for values in matched_ious.values() for value in values]),
        "mean_mask_dice": aggregate([value for values in matched_dices.values() for value in values]),
        "matched_instances": sum(len(values) for values in matched_ious.values()),
    }
    result = {
        "split": split,
        "checkpoint": str(model_path),
        "score_threshold": score_threshold,
        "aggregate": aggregate_metrics,
        "per_class": category_metrics,
        "predictions": saved_predictions,
    }
    output_path = output_dir / f"mask_metrics_{split}.json"
    with open(output_path, "w") as file:
        json.dump(result, file)
    print(f"Saved mask predictions and metrics to {output_path}")
    return {key: value for key, value in result.items() if key != "predictions"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/processed_data/coco_format/maskrcnn")
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/train_script/maskrcnn_project/run_output/maskrcnn_resnet50_fpn_best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/train_script/maskrcnn_project/run_output/mask_metrics")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    first_dataset = SurgicalMaskRCNNDataset(args.root, split=args.splits[0])
    model = get_model(len(first_dataset.cat_ids) + 1, device, weights=None)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint from {args.model_path}")

    all_results = {}
    for split in args.splits:
        all_results[split] = evaluate_split(
            split,
            model,
            device,
            args.root,
            args.output_dir,
            args.score_threshold,
            args.model_path,
        )
    summary_path = args.output_dir / "mask_metrics_summary.json"
    with open(summary_path, "w") as file:
        json.dump(all_results, file, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
