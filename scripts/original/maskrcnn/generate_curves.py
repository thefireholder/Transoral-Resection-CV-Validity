#!/usr/bin/env python3
"""
Generate precision-recall and precision-confidence curves for train/val/test.

Two modes:
  1. Inference mode (default): loads model weights, runs inference, saves raw
     predictions JSON, then computes and saves curves.
  2. From-saved mode (--from-saved): skips inference entirely, loads the
     previously saved raw_predictions_{split}.json files, and regenerates
     curves. No GPU or model needed.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/projects/illinois/cimed/bts/gayed')
COCO_ROOT = ROOT / 'data/processed_data/coco_format/maskrcnn'
RUN_OUTPUT = ROOT / 'data/train_script/maskrcnn_project/run_output'
OUT_DIR = RUN_OUTPUT / 'curves'

sys.path.append(str(ROOT / 'data/train_script/maskrcnn_project'))
from train_maskrcnn import (
    SurgicalMaskRCNNDataset, collate_fn, get_model, set_seed,
)


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_inference(split_name, model, device):
    dataset = SurgicalMaskRCNNDataset(COCO_ROOT, split=split_name)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    gt_by_image = {}
    for image_id in dataset.image_ids:
        anns = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id]))
        gt_boxes, gt_labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(dataset.cat_id_to_idx[ann['category_id']])
        gt_by_image[image_id] = {'boxes': gt_boxes, 'labels': gt_labels}

    raw_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            imgs = [img.to(device) for img in images]
            outputs = model(imgs)
            for target, output in zip(targets, outputs):
                image_id = int(target['image_id'].item())
                raw_predictions.append({
                    'image_id': image_id,
                    'boxes': output['boxes'].cpu().tolist(),
                    'labels': output['labels'].cpu().tolist(),
                    'scores': output['scores'].cpu().tolist(),
                })
            if (batch_idx + 1) % 10 == 0:
                done = min((batch_idx + 1) * 2, len(dataset))
                print(f'  {split_name}: {done}/{len(dataset)} images', flush=True)

    return gt_by_image, raw_predictions


def save_raw_predictions(split_name, gt_by_image, raw_predictions):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'split': split_name,
        'gt_by_image': {str(k): v for k, v in gt_by_image.items()},
        'predictions': raw_predictions,
    }
    path = OUT_DIR / f'raw_predictions_{split_name}.json'
    with open(path, 'w') as f:
        json.dump(payload, f)
    print(f'  Saved raw predictions -> {path}')
    return path


def load_raw_predictions(split_name):
    path = OUT_DIR / f'raw_predictions_{split_name}.json'
    if not path.exists():
        raise FileNotFoundError(
            f'Raw predictions not found: {path}\n'
            'Run without --from-saved first to generate them.'
        )
    with open(path) as f:
        payload = json.load(f)
    gt_by_image = {int(k): v for k, v in payload['gt_by_image'].items()}
    return gt_by_image, payload['predictions']


def compute_curves(gt_by_image, raw_predictions, iou_threshold=0.5):
    labels = sorted({
        label
        for image in gt_by_image.values()
        for label in image['labels']
    } | {
        label
        for prediction in raw_predictions
        for label in prediction['labels']
    })
    curves_by_label = {}

    for label in labels:
        total_gt = sum(
            gt_label == label
            for image in gt_by_image.values()
            for gt_label in image['labels']
        )
        label_preds = []

        for pred in raw_predictions:
            image_id = pred['image_id']
            image_gt = gt_by_image.get(image_id, {})
            gt_boxes = image_gt.get('boxes', [])
            gt_labels = image_gt.get('labels', [])
            gt_used = [False] * len(gt_boxes)
            items = [
                (box, score)
                for box, pred_label, score in zip(
                    pred['boxes'], pred['labels'], pred['scores']
                )
                if pred_label == label
            ]
            items.sort(key=lambda item: item[1], reverse=True)

            for box, score in items:
                matched = False
                for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_used[idx] or gt_label != label:
                        continue
                    if iou(box, gt_box) >= iou_threshold:
                        gt_used[idx] = True
                        matched = True
                        break
                label_preds.append((float(score), 1 if matched else 0))

        label_preds.sort(key=lambda item: item[0], reverse=True)
        tp = fp = 0
        precision_hist, recall_hist, conf_hist = [], [], []
        for score, is_tp in label_preds:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precision_hist.append(tp / (tp + fp) if (tp + fp) else 0.0)
            recall_hist.append(tp / max(1, total_gt))
            conf_hist.append(score)

        curves_by_label[label] = {
            'confidence': conf_hist,
            'precision': precision_hist,
            'recall': recall_hist,
        }

    return curves_by_label


def load_category_names():
    annotation_path = COCO_ROOT / 'annotations' / 'instances_train.json'
    with open(annotation_path) as f:
        categories = json.load(f)['categories']
    return {category['id']: category['name'] for category in categories}


def save_curves(curves_by_split, category_names):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ('Precision-Recall', 'Recall', 'Precision', 'recall', 'precision', 'precision_recall'),
        ('Precision-Confidence', 'Confidence Threshold', 'Precision', 'confidence', 'precision', 'precision_confidence'),
        ('Recall-Confidence', 'Confidence Threshold', 'Recall', 'confidence', 'recall', 'recall_confidence'),
    ]
    for split_name, curves_by_label in curves_by_split.items():
        for title, xlabel, ylabel, key_x, key_y, filename in plot_specs:
            fig, ax = plt.subplots(figsize=(11, 7))
            for label, curve in curves_by_label.items():
                label_name = category_names.get(label, f'label_{label}')
                ax.plot(curve[key_x], curve[key_y], label=label_name, linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} Curves - {split_name}')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            output_path = OUT_DIR / f'{filename}_{split_name}.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            print(f'  Saved {output_path}')

        for label, curve in curves_by_label.items():
            label_name = category_names.get(label, f'label_{label}')
            safe_name = ''.join(
                char if char.isalnum() else '_' for char in label_name.lower()
            ).strip('_')
            csv_path = OUT_DIR / f'{split_name}_{label:02d}_{safe_name}_curve.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['confidence', 'precision', 'recall'])
                for conf, prec, rec in zip(
                    curve['confidence'], curve['precision'], curve['recall']
                ):
                    writer.writerow([conf, prec, rec])
            print(f'  Saved {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-saved', action='store_true',
                        help='Load raw_predictions_{split}.json instead of running inference.')
    parser.add_argument('--model-path', type=Path,
                        default=RUN_OUTPUT / 'maskrcnn_resnet50_fpn_best.pth',
                        help='Path to model weights (ignored with --from-saved).')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    if not args.from_saved and not args.model_path.exists():
        fallback = RUN_OUTPUT / 'maskrcnn_resnet50_fpn_final.pth'
        if fallback.exists():
            print(f'Best checkpoint not found; falling back to {fallback}')
            args.model_path = fallback
        else:
            raise FileNotFoundError(f'No model weights at {args.model_path} or {fallback}')

    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.from_saved:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        num_classes = len(SurgicalMaskRCNNDataset(COCO_ROOT, split='train').cat_ids) + 1
        model = get_model(num_classes, device, weights=None)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        print(f'Loaded model from {args.model_path}')
    else:
        model = device = None
        print('--from-saved: skipping model inference')

    category_names = load_category_names()
    curves_by_split = {}
    for split_name in args.splits:
        print(f'\nProcessing split: {split_name}')
        if args.from_saved:
            gt_by_image, raw_predictions = load_raw_predictions(split_name)
        else:
            gt_by_image, raw_predictions = run_inference(split_name, model, device)
            save_raw_predictions(split_name, gt_by_image, raw_predictions)
        curves_by_split[split_name] = compute_curves(
            gt_by_image, raw_predictions, iou_threshold=args.iou_threshold
        )

    print('\nSaving plots and CSVs...')
    save_curves(curves_by_split, category_names)
    print(f'\nAll outputs written to {OUT_DIR}')


if __name__ == '__main__':
    main()
