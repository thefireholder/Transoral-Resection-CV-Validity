#!/usr/bin/env python3
"""Script 17: Instance-level AP (mAP@0.5, mAP@0.5:0.95) for the unified model.

The UNet is a semantic segmenter, so instances are derived the standard way
for semantic models: 8-connected components of the class mask, for both
prediction and ground truth. Each predicted component is scored by the mean
softmax probability of its class over its pixels. AP is COCO-style: greedy
score-ordered matching on mask IoU, 101-point interpolated PR curve, averaged
over IoU thresholds 0.50:0.05:0.95 for AP@0.5:0.95.

Conventions:
- Components smaller than MIN_COMPONENT_PX (on the 512x512 eval grid) are
  discarded as noise, in both GT and prediction.
- AP is computed per class over the combined test split; classes with zero GT
  instances are excluded (AP undefined).

Output: outputs/evaluation_unified/instance_ap.json
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, UnifiedClassConfig, ModelConfig
from src.models.segmentation import SegmentationModel
from src.utils.device import get_device

SIZE = 512
MIN_COMPONENT_PX = 20
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)  # 0.50 ... 0.95


def components(mask_bool):
    """List of pixel-index arrays, one per 8-connected component >= min size."""
    n, lab = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
    out = []
    flat = lab.ravel()
    for k in range(1, n):
        idx = np.flatnonzero(flat == k)
        if idx.size >= MIN_COMPONENT_PX:
            out.append(idx)
    return out


def ap_101(recall, precision):
    """COCO 101-point interpolated AP from unsorted PR points (score-ordered)."""
    # precision envelope (monotone non-increasing from the right)
    mprec = np.concatenate(([0.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, pts, side="left")
    return float(np.mean(np.where(idx < len(mprec), mprec[np.minimum(idx, len(mprec) - 1)], 0.0)))


def main():
    config = get_config()
    cfg = UnifiedClassConfig()
    device = get_device()
    n_cls = cfg.num_classes

    ckpt = torch.load(config.paths.unified_checkpoints_dir / "best_model.pth",
                      map_location=device, weights_only=False)
    model = SegmentationModel(ModelConfig(out_channels=n_cls))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    split = json.load(open(config.paths.unified_split_file))
    entries = split["test"]
    print(f"Computing instance AP on test split ({len(entries)} frames)...")

    # Per class: detections as (score, image_i, iou_row vs that image's GT comps),
    # and GT instance counts per image.
    dets = {c: [] for c in range(1, n_cls)}
    n_gt = {c: 0 for c in range(1, n_cls)}

    with torch.no_grad():
        for i, fname in enumerate(entries):
            img = cv2.imread(str(config.paths.unified_frames_dir / fname))
            gt = cv2.imread(str(config.paths.unified_masks_dir / fname.replace(".jpg", ".png")),
                            cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE, SIZE))
            gt = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            pred = probs.argmax(axis=0).astype(np.uint8)

            for c in range(1, n_cls):
                gt_comps = components(gt == c) if (gt == c).any() else []
                pr_comps = components(pred == c) if (pred == c).any() else []
                n_gt[c] += len(gt_comps)
                if not pr_comps:
                    continue
                pflat = probs[c].ravel()
                gt_sets = [(set(g.tolist()), g.size) for g in gt_comps]
                for p in pr_comps:
                    score = float(pflat[p].mean())
                    pset = set(p.tolist())
                    ious = np.zeros(len(gt_comps))
                    for j, (gset, gsize) in enumerate(gt_sets):
                        inter = len(pset & gset)
                        if inter:
                            ious[j] = inter / (p.size + gsize - inter)
                    dets[c].append((score, i, ious))
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(entries)} frames")

    results = {"conventions": {
        "instances": "8-connected components of the semantic mask (GT and pred)",
        "min_component_px": MIN_COMPONENT_PX,
        "score": "mean softmax probability of the class over the component",
        "ap": "COCO 101-point interpolation, IoU thresholds 0.50:0.05:0.95",
        "split": "combined test (158 frames)",
    }, "classes": {}}

    for c in range(1, n_cls):
        if n_gt[c] == 0:
            continue
        d = sorted(dets[c], key=lambda t: -t[0])
        aps = []
        for t in IOU_THRESHOLDS:
            matched = {}  # image_i -> set of matched gt indices
            tp = np.zeros(len(d)); fp = np.zeros(len(d))
            for k, (score, img_i, ious) in enumerate(d):
                cand = [(v, j) for j, v in enumerate(ious)
                        if v >= t and j not in matched.get(img_i, set())]
                if cand:
                    _, j = max(cand)
                    matched.setdefault(img_i, set()).add(j)
                    tp[k] = 1
                else:
                    fp[k] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            rec = ctp / n_gt[c]
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            aps.append(ap_101(rec, prec))
        results["classes"][cfg.id_to_class[c]] = {
            "ap50": round(aps[0], 4),
            "ap50_95": round(float(np.mean(aps)), 4),
            "n_gt_instances": n_gt[c],
            "n_pred_instances": len(d),
        }

    vals = results["classes"].values()
    results["aggregate"] = {
        "map50": round(float(np.mean([v["ap50"] for v in vals])), 4),
        "map50_95": round(float(np.mean([v["ap50_95"] for v in vals])), 4),
        "n_classes": len(results["classes"]),
    }

    out = config.paths.outputs_dir / "evaluation_unified" / "instance_ap.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'class':<28} {'AP@0.5':>8} {'AP@0.5:0.95':>12} {'GT inst':>8} {'pred inst':>10}")
    for cls, v in results["classes"].items():
        print(f"{cls:<28} {v['ap50']:>8.4f} {v['ap50_95']:>12.4f} "
              f"{v['n_gt_instances']:>8} {v['n_pred_instances']:>10}")
    a = results["aggregate"]
    print(f"\nmAP@0.5: {a['map50']:.4f}   mAP@0.5:0.95: {a['map50_95']:.4f} "
          f"(over {a['n_classes']} classes with GT)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
