"""
seglib.py -- shared helpers for mask-level evaluation, used by every script in
scripts/ so that all models are scored with exactly the same code.

Uniform prediction file format (data/predictions/<model>_test.json):

    {
      "model": "yolo",
      "split": "test",
      "score_threshold": 0.05,
      "images": [
        {
          "file_name": "JML619-images__frame_33806.jpg",
          "height": 1080, "width": 1920,
          "predictions": [
            {"class": "soft palate", "score": 0.91,
             "rle": {"size": [1080, 1920], "counts": "<coco rle string>"}},
            ...
          ]
        }, ...
      ]
    }

Ground truth is read straight from the COCO json of the split.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------------------
# Paths. On the cluster nothing to set. On another machine (laptop) export
#   TORS_SHARED=/path/that/mirrors/u/sl257/shared_data
# i.e. a folder containing processed_data/{coco_format,yolo_format}/1200images
# (and train_script/... only if you need the original checkpoints/logs).
# ---------------------------------------------------------------------------
SHARED = Path(os.environ.get("TORS_SHARED", "/u/sl257/shared_data"))
DATA_ROOT = SHARED / "processed_data"
COCO_1200 = DATA_ROOT / "coco_format" / "1200images"
YOLO_1200 = DATA_ROOT / "yolo_format" / "1200images"

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # COCO / ultralytics convention
PR_RECALL_GRID = np.linspace(0, 1, 1000)     # ultralytics PR-curve x axis
AP_METHOD = "coco"                           # "coco" | "ultralytics"; see _compute_ap


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------
def rle_encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def rle_area(rle: dict) -> float:
    return float(mask_utils.area(rle))


def coco_ann_to_rle(ann: dict, h: int, w: int) -> dict | None:
    """Polygon / RLE COCO annotation -> RLE dict (None if empty / invalid)."""
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        polys = [p for p in seg if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
        if not polys:
            return None
        rles = mask_utils.frPyObjects(polys, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        if isinstance(seg["counts"], list):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
    else:
        return None
    if mask_utils.area(rle) == 0:
        return None
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [h, w], "counts": counts}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_coco_gt(split: str = "test", coco_root: Path = COCO_1200) -> dict:
    """Returns {"classes": [names in category-id order],
                "images": {file_name: {"height","width","gts":[{"class","rle"}]}}}"""
    with open(coco_root / "annotations" / f"instances_{split}.json") as f:
        coco = json.load(f)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {}
    for im in coco["images"]:
        images[im["id"]] = {"file_name": im["file_name"], "height": im["height"],
                            "width": im["width"], "gts": []}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        im = images[ann["image_id"]]
        rle = coco_ann_to_rle(ann, im["height"], im["width"])
        if rle is None:
            continue
        im["gts"].append({"class": cat_name[ann["category_id"]], "rle": rle})
    return {
        "classes": [cat_name[i] for i in sorted(cat_name)],
        "images": {v["file_name"]: v for v in images.values()},
    }


# ---------------------------------------------------------------------------
# Prediction file I/O
# ---------------------------------------------------------------------------
def save_predictions(path: Path, model: str, split: str, score_threshold: float, images: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model": model, "split": split, "score_threshold": score_threshold,
                   "images": images}, f)


def load_predictions(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Matching + metrics
# ---------------------------------------------------------------------------
def _pairwise_iou(pred_rles, gt_rles) -> np.ndarray:
    if not pred_rles or not gt_rles:
        return np.zeros((len(pred_rles), len(gt_rles)))
    return np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)))


def evaluate_predictions(pred: dict, gt: dict, classes: list[str] | None = None,
                         ap_method: str | None = None) -> dict:
    """
    Score a uniform prediction file against COCO GT.

    Per class, predictions are sorted by score and greedily matched to unused
    GT instances of the same class by mask IoU (same rule as
    eval_saved_mask_metrics.py for Mask R-CNN). A match at IoU>=0.5 counts as
    TP for AP50 and contributes its IoU / Dice to the class mean; TP at each
    of the 10 COCO thresholds gives AP50:95 (ultralytics-style 101-pt interp).

    Returns:
      {
        "classes": [...],
        "per_class": {cls: {"AP50","AP50_95","precision","recall","iou","dice",
                            "n_gt","n_pred","n_matched"}},
        "aggregate": {"mAP50","mAP50_95","precision","recall","iou","dice"},
        "pr_curves": {cls: {"recall": [1000], "precision": [1000]}},  # at IoU 0.5
      }
    precision / recall in per_class are at IoU 0.5 and at the single
    confidence threshold that maximises the (smoothed) mean-over-classes F1,
    exactly the ultralytics Mask(P, R) convention; aggregate["conf_at_max_f1"]
    records that threshold. Use pr_curves for a threshold-free view.
    """
    if classes is None:
        classes = gt["classes"]

    tp_rows = defaultdict(list)      # cls -> list of (score, tp@10thr)
    n_gt = defaultdict(int)
    match_iou = defaultdict(list)
    match_dice = defaultdict(list)

    for im in pred["images"]:
        g = gt["images"].get(im["file_name"])
        if g is None:
            raise KeyError(f"{im['file_name']} not in GT split")
        gts_by_cls = defaultdict(list)
        for x in g["gts"]:
            gts_by_cls[x["class"]].append(x["rle"])
            n_gt[x["class"]] += 1
        preds_by_cls = defaultdict(list)
        for p in im["predictions"]:
            preds_by_cls[p["class"]].append(p)

        for cls, preds in preds_by_cls.items():
            preds = sorted(preds, key=lambda p: -p["score"])
            gt_rles = gts_by_cls.get(cls, [])
            ious = _pairwise_iou([p["rle"] for p in preds], gt_rles)
            used = np.zeros(len(gt_rles), dtype=bool)
            for i, p in enumerate(preds):
                tp = np.zeros(len(IOU_THRESHOLDS), dtype=np.uint8)
                if len(gt_rles):
                    cand = np.where(~used)[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        best = ious[i, j]
                        if best >= 0.5:
                            used[j] = True
                            tp = (best >= IOU_THRESHOLDS).astype(np.uint8)
                            a_p, a_g = rle_area(p["rle"]), rle_area(gt_rles[j])
                            inter = best * (a_p + a_g) / (1 + best)  # from IoU = I/(A+B-I)
                            match_iou[cls].append(float(best))
                            match_dice[cls].append(float(2 * inter / (a_p + a_g)))
                tp_rows[cls].append((p["score"], tp))

    # --- precision / recall vs confidence on a common grid, per class ---------
    conf_grid = np.linspace(0, 1, 1000)
    p_curves, r_curves, f1_curves = {}, {}, {}
    for cls in classes:
        rows = tp_rows.get(cls, [])
        ngt = n_gt.get(cls, 0)
        if ngt == 0 and not rows:
            continue
        if rows:
            rows.sort(key=lambda r: -r[0])
            conf = np.array([r[0] for r in rows])
            tp50 = np.array([r[1][0] for r in rows], dtype=float)
            tpc, fpc = tp50.cumsum(), (1 - tp50).cumsum()
            rec = tpc / max(ngt, 1)
            prec = tpc / np.maximum(tpc + fpc, 1e-9)
            # value at confidence threshold c = value at the last prediction with score >= c
            pc = np.interp(-conf_grid, -conf, prec, left=prec[-1] if len(prec) else 0, right=1.0)
            rc = np.interp(-conf_grid, -conf, rec, left=rec[-1] if len(rec) else 0, right=0.0)
        else:
            pc = np.zeros_like(conf_grid); rc = np.zeros_like(conf_grid)
        p_curves[cls], r_curves[cls] = pc, rc
        f1_curves[cls] = 2 * pc * rc / np.maximum(pc + rc, 1e-9)
    if f1_curves:
        mean_f1 = np.mean(np.stack(list(f1_curves.values())), axis=0)
        i_best = int(np.argmax(_smooth(mean_f1, 0.1)))   # ultralytics picks argmax of smoothed mean F1
        conf_best = float(conf_grid[i_best])
    else:
        i_best, conf_best = 0, 0.0

    per_class, pr_curves = {}, {}
    for cls in classes:
        rows = tp_rows.get(cls, [])
        ngt = n_gt.get(cls, 0)
        if ngt == 0 and not rows:
            continue  # class absent from this split and never predicted
        if rows:
            rows.sort(key=lambda r: -r[0])
            tp = np.stack([r[1] for r in rows]).astype(float)   # (n_pred, 10)
            fp = 1 - tp
            tpc, fpc = tp.cumsum(0), fp.cumsum(0)
            recall = tpc / max(ngt, 1)
            precision = tpc / np.maximum(tpc + fpc, 1e-9)
            ap = np.array([_compute_ap(recall[:, k], precision[:, k], ap_method)
                           for k in range(len(IOU_THRESHOLDS))])
            # PR curve at IoU 0.5 on fixed recall grid: monotone precision
            # envelope, exactly what ultralytics plots in MaskPR_curve.png
            mrec = np.concatenate(([0.0], recall[:, 0], [1.0]))
            mpre = np.concatenate(([1.0], precision[:, 0], [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            prec_curve = np.interp(PR_RECALL_GRID, mrec, mpre)
        else:
            ap = np.zeros(len(IOU_THRESHOLDS))
            prec_curve = np.zeros_like(PR_RECALL_GRID)
        ious, dices = match_iou.get(cls, []), match_dice.get(cls, [])
        per_class[cls] = {
            "AP50": float(ap[0]) if ngt else None,
            "AP50_95": float(ap.mean()) if ngt else None,
            "precision": float(p_curves[cls][i_best]) if rows else None,
            "recall": float(r_curves[cls][i_best]) if ngt else None,
            "iou": float(np.mean(ious)) if ious else None,
            "dice": float(np.mean(dices)) if dices else None,
            "n_gt": int(ngt), "n_pred": len(rows), "n_matched": len(ious),
        }
        pr_curves[cls] = {"recall": PR_RECALL_GRID.tolist(), "precision": prec_curve.tolist()}

    def _mean(key, only_with_gt=True):
        vals = [v[key] for v in per_class.values()
                if v[key] is not None and (v["n_gt"] > 0 or not only_with_gt)]
        return float(np.mean(vals)) if vals else None

    all_iou = [x for v in match_iou.values() for x in v]
    all_dice = [x for v in match_dice.values() for x in v]
    aggregate = {
        "mAP50": _mean("AP50"),
        "mAP50_95": _mean("AP50_95"),
        "precision": _mean("precision"),
        "recall": _mean("recall"),
        # instance-pooled, same as Mask R-CNN eval_saved_mask_metrics.py and SAM3
        "iou": float(np.mean(all_iou)) if all_iou else None,
        "dice": float(np.mean(all_dice)) if all_dice else None,
        "conf_at_max_f1": conf_best,
    }
    return {"classes": list(per_class), "per_class": per_class,
            "aggregate": aggregate, "pr_curves": pr_curves}


def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f (ultralytics.utils.metrics.smooth)."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def _compute_ap(recall: np.ndarray, precision: np.ndarray, method: str | None = None) -> float:
    """
    Average precision from a (recall, precision) sequence sorted by score.

    method="coco"        : pycocotools rule -- monotone precision envelope,
                           sampled at 101 recall points, 0 beyond max recall.
                           (Mask R-CNN and MONAI reported numbers use this.)
    method="ultralytics" : ultralytics.utils.metrics.compute_ap -- same
                           envelope but linearly interpolated to (recall=1, p=0),
                           which inflates AP when recall saturates below 1.
                           (YOLO and SAM3 reported numbers use this.)
    """
    method = method or AP_METHOD
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    if method == "ultralytics":
        return float(np.trapezoid(np.interp(x, mrec, mpre), x))
    if method == "coco":
        rec = np.asarray(recall)
        env = np.flip(np.maximum.accumulate(np.flip(np.asarray(precision)))) if len(rec) else rec
        q = np.zeros_like(x)
        idx = np.searchsorted(rec, x, side="left")
        ok = idx < len(rec)
        q[ok] = env[idx[ok]]
        return float(q.mean())
    raise ValueError(method)


def save_pr_curves(path: Path, model: str, result: dict):
    """data/pr_curves/<model>.json : {"model", "recall":[1000], "classes":{cls:{"precision":[1000],"AP50":..}}}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"model": model, "iou_threshold": 0.5, "recall": PR_RECALL_GRID.tolist(), "classes": {}}
    for cls, c in result["pr_curves"].items():
        out["classes"][cls] = {"precision": c["precision"], "AP50": result["per_class"][cls]["AP50"]}
    with open(path, "w") as f:
        json.dump(out, f)
