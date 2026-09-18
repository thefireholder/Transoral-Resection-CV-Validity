#!/usr/bin/env python3
"""Script 14: Extended evaluation of the unified (JSR+JML619) model.

Same conventions as scripts/10_evaluate_jml619.py (documented there and in the
report). Additionally evaluates the test split per source case (test_jsr,
test_jml) — the per-case view is where cross-case behavior shows up, since
several classes exist in only one case's vocabulary.

Outputs: outputs/evaluation_unified/results.json + qualitative overlays.
"""
import sys
import json
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, UnifiedClassConfig, ModelConfig
from src.models.segmentation import SegmentationModel
from src.utils.device import get_device
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

SIZE = 512
SPACING = (1080 / SIZE, 1920 / SIZE)
NSD_TOL_PX = 15.0
PRESENCE_PX = 50
EVALUABLE_MIN_FRAMES = 5


def to_onehot(labels: np.ndarray, n: int) -> torch.Tensor:
    t = torch.from_numpy(labels.astype(np.int64))[None]
    oh = torch.zeros(1, n, *labels.shape)
    oh.scatter_(1, t.unsqueeze(1), 1)
    return oh


def evaluate_entries(entries, model, device, cfg, frames_dir, masks_dir):
    n = cfg.num_classes
    per_frame_dice = {c: [] for c in range(1, n)}
    hd95 = {c: [] for c in range(1, n)}
    nsd = {c: [] for c in range(1, n)}
    missed = {c: 0 for c in range(1, n)}
    pooled = np.zeros((n, 3), dtype=np.int64)
    presence = {c: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for c in range(1, n)}
    gt_frames = {c: 0 for c in range(1, n)}
    confusion = np.zeros((n, n), dtype=np.int64)
    frame_scores = []

    with torch.no_grad():
        for fname in entries:
            img = cv2.imread(str(frames_dir / fname))
            gt = cv2.imread(str(masks_dir / fname.replace(".jpg", ".png")),
                            cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE, SIZE))
            gt = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
            pred = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)

            confusion += np.bincount(
                (gt.astype(np.int64) * n + pred).ravel(), minlength=n * n
            ).reshape(n, n)

            gt_oh, pr_oh = to_onehot(gt, n), to_onehot(pred, n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hd = compute_hausdorff_distance(
                    pr_oh, gt_oh, include_background=False,
                    percentile=95, spacing=SPACING).numpy()[0]
                sd = compute_surface_dice(
                    pr_oh, gt_oh, class_thresholds=[NSD_TOL_PX] * (n - 1),
                    include_background=False, spacing=SPACING).numpy()[0]

            frame_dice = []
            for c in range(1, n):
                g, p = gt == c, pred == c
                tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
                pooled[c] += (tp, fp, fn)
                g_present, p_present = g.sum() >= PRESENCE_PX, p.sum() >= PRESENCE_PX
                if g_present:
                    gt_frames[c] += 1
                    presence[c]["tp" if p_present else "fn"] += 1
                else:
                    presence[c]["fp" if p_present else "tn"] += 1
                if g.any():
                    d = 2 * tp / (2 * tp + fp + fn)
                    per_frame_dice[c].append(d)
                    frame_dice.append(d)
                    if p.any():
                        if np.isfinite(hd[c - 1]):
                            hd95[c].append(float(hd[c - 1]))
                        if np.isfinite(sd[c - 1]):
                            nsd[c].append(float(sd[c - 1]))
                    else:
                        missed[c] += 1
            if frame_dice:
                frame_scores.append((float(np.mean(frame_dice)), fname))

    classes = {}
    for c in range(1, n):
        tp, fp, fn = pooled[c]
        d = per_frame_dice[c]
        classes[cfg.id_to_class[c]] = {
            "gt_frames": gt_frames[c],
            "evaluable": gt_frames[c] >= EVALUABLE_MIN_FRAMES,
            "dice_mean": float(np.mean(d)) if d else None,
            "dice_median": float(np.median(d)) if d else None,
            "dice_per_frame": [round(v, 4) for v in d],
            "dice_pooled": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "iou_pooled": float(tp / (tp + fp + fn)) if (tp + fp + fn) else None,
            "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            "recall": float(tp / (tp + fn)) if (tp + fn) else None,
            "hd95_median_px": float(np.median(hd95[c])) if hd95[c] else None,
            "nsd15_mean": float(np.mean(nsd[c])) if nsd[c] else None,
            "missed_frames": missed[c],
            "presence": presence[c],
        }
    ev = [v["dice_mean"] for v in classes.values() if v["evaluable"] and v["dice_mean"] is not None]
    al = [v["dice_mean"] if v["dice_mean"] is not None else 0.0 for v in classes.values()]
    nsd_ev = [v["nsd15_mean"] for v in classes.values() if v["evaluable"] and v["nsd15_mean"] is not None]
    summary = {
        "n_frames": len(entries),
        "mean_dice_evaluable": float(np.mean(ev)) if ev else None,
        "n_evaluable": len(ev),
        "mean_dice_all_classes": float(np.mean(al)) if al else None,
        "mean_nsd_evaluable": float(np.mean(nsd_ev)) if nsd_ev else None,
    }
    frame_scores.sort()
    picks = {}
    if frame_scores:
        picks = {"worst": frame_scores[0],
                 "median": frame_scores[len(frame_scores) // 2],
                 "best": frame_scores[-1]}
    return {"summary": summary, "classes": classes,
            "confusion": confusion.tolist(),
            "qualitative": {k: {"file": f, "score": round(s, 4)}
                            for k, (s, f) in picks.items()}}


def main():
    config = get_config()
    cfg = UnifiedClassConfig()
    device = get_device()

    ckpt = torch.load(config.paths.unified_checkpoints_dir / "best_model.pth",
                      map_location=device, weights_only=False)
    model = SegmentationModel(ModelConfig(out_channels=cfg.num_classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded unified best_model.pth (best val dice {ckpt['best_dice']:.4f})")

    split = json.load(open(config.paths.unified_split_file))
    subsets = {
        "val": split["val"],
        "test": split["test"],
        "test_jsr": [e for e in split["test"] if e.startswith("jsr_")],
        "test_jml": [e for e in split["test"] if e.startswith("jml_")],
    }

    out_dir = config.paths.outputs_dir / "evaluation_unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"conventions": {
        "resolution": SIZE, "spacing_to_original_px": SPACING,
        "nsd_tolerance_px": NSD_TOL_PX, "presence_threshold_px": PRESENCE_PX,
        "evaluable_min_gt_frames": EVALUABLE_MIN_FRAMES,
        "best_val_dice": float(ckpt["best_dice"]),
    }}
    fmt = lambda x, spec=".3f": format(x, spec) if x is not None else "  n/a"
    for name, entries in subsets.items():
        print(f"\nEvaluating {name} ({len(entries)} frames)...")
        results[name] = evaluate_entries(
            entries, model, device, cfg,
            config.paths.unified_frames_dir, config.paths.unified_masks_dir)
        s = results[name]["summary"]
        print(f"  mean Dice (evaluable, n={s['n_evaluable']}): {fmt(s['mean_dice_evaluable'])}"
              f"   NSD@15 {fmt(s['mean_nsd_evaluable'])}")
        for cls, v in results[name]["classes"].items():
            if v["dice_mean"] is not None:
                print(f"    {cls:<28} dice {fmt(v['dice_mean'])}  P {fmt(v['precision'])}  "
                      f"R {fmt(v['recall'])}  gtframes {v['gt_frames']}"
                      f"{'' if v['evaluable'] else '  [not evaluable]'}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
