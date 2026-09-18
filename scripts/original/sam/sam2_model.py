import math
import warnings
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import numpy as np
import torch

# Adjust imports to match your SAM2 repo layout
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # Pass PR curve data for interactive plotting (class names stored at model level)
        # Transpose py to match other curves: y[class][point] format
        on_plot(save_dir, {"type": "pr_curve", "x": px.tolist(), "y": py.T.tolist(), "ap": ap.tolist()})


def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # Pass metric-confidence curve data for interactive plotting (class names stored at model level)
        on_plot(save_dir, {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()})


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid # if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []
    # Average precision, precision and recall curves

    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# If your COCO JSON uses RLE segmentation, you'll need pycocotools.
# pip install pycocotools
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


# -----------------------------
# Utilities: masks, boxes, metrics
# -----------------------------

from pycocotools import mask as mask_utils
import numpy as np


def coco_ann_to_binary_mask(coco, ann, h: int, w: int) -> np.ndarray:
    """
    Robustly convert a COCO annotation to a binary mask (H, W).
    Handles polygon, uncompressed RLE, compressed RLE, and skips invalid cases.
    """
    seg = ann.get("segmentation", None)

    if seg is None or seg == []:
        # No segmentation → empty mask
        return np.zeros((h, w), dtype=np.uint8)

    try:
        # Case 1: Polygon (list of lists)
        if isinstance(seg, list):
            # Polygon(s)
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles)

        # Case 2: RLE dict
        elif isinstance(seg, dict) and "counts" in seg:
            # Already RLE (compressed or uncompressed)
            rle = seg

        else:
            raise ValueError(f"Unsupported segmentation type: {type(seg)}")

        mask = mask_utils.decode(rle)

        # mask can be (H, W, 1)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return (mask > 0).astype(np.uint8)

    except Exception as e:
        # Log and return empty mask instead of crashing
        print(f"[WARN] Failed to decode ann_id={ann.get('id', 'N/A')}: {e}")
        return np.zeros((h, w), dtype=np.uint8)



def xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = box_xywh.astype(float)
    return np.array([x, y, x + w, y + h], dtype=float)


def pad_box_xyxy(box_xyxy: np.ndarray, pad_frac: float, img_w: int, img_h: int) -> np.ndarray:
    """
    Pad a box by pad_frac of its size in both directions and clamp to image bounds.
    """
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad_frac, bh * pad_frac
    x1p, y1p = x1 - px, y1 - py
    x2p, y2p = x2 + px, y2 + py
    x1p = max(0.0, x1p)
    y1p = max(0.0, y1p)
    x2p = min(float(img_w - 1), x2p)
    y2p = min(float(img_h - 1), y2p)
    return np.array([x1p, y1p, x2p, y2p], dtype=float)


def mask_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + eps))


def mask_dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2 * inter) / (denom + eps))


def coco_iou_thresholds() -> np.ndarray:
    # COCO: 0.50:0.05:0.95 (10 thresholds)
    return np.round(np.arange(0.50, 0.96, 0.05), 2)


# -----------------------------
# SAM2 Predictor interface (you implement)
# -----------------------------

class SAM2PredictorInterface:
    """
    Implement `predict_mask(image_rgb_uint8, box_xyxy_float)` -> binary mask uint8 {0,1} (H, W).
    """
    def predict_mask(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
class SAM21Predictor(SAM2PredictorInterface):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = device

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        initialize_config_dir(
            config_dir="/projects/illinois/cimed/bts/gayed/data/train_script/SAM2_project/sam2/sam2/configs",
            job_name="sam2_eval",
        )

        # SAM 2.1 build (NO config_file needed)
        self.model = build_sam2(
            config_file="sam2.1/sam2.1_hiera_l",
            ckpt_path=checkpoint_path,
            device=device,
        )
        self.model.eval()

        self.predictor = SAM2ImagePredictor(self.model)

    @torch.no_grad()
    def predict_mask(
        self,
        image_rgb: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> np.ndarray:
        """
        image_rgb: (H, W, 3), uint8 RGB
        box_xyxy: (4,) float pixel coords
        """
        self.predictor.set_image(image_rgb)

        masks, scores, logits = self.predictor.predict(
            box=box_xyxy.astype(np.float32)[None, :],
            multimask_output=False,
        )

        mask = masks[0] if masks.ndim == 3 else masks
        return (mask > 0).astype(np.uint8)
    
# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class EvalConfig:
    dataset_root: Path  # JS_R_Tonsil_coco_merged_v3_vv4/
    pad_frac: float = 0.05
    iou_thrs: np.ndarray = field(default_factory=coco_iou_thresholds)
    max_instances: Optional[int] = None  # set for quick debug (e.g., 200)


def load_image_rgb(path: Path) -> np.ndarray:
    # Keep dependencies minimal. Uses PIL.
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


def evaluate_sam2_on_coco_val(
    cfg: EvalConfig,
    predictor: SAM2PredictorInterface,
    result_path: Path,
) -> Dict[str, object]:
    
    ann_path = cfg.dataset_root / "annotations" / "instances_test.json"
    img_dir = cfg.dataset_root / "images" / "test"

    coco = COCO(str(ann_path))

    # COCO categories mapping
    cats = coco.loadCats(coco.getCatIds())
    names = {c["id"]: c.get("name", str(c["id"])) for c in cats}  # category_id -> name

    # We'll collect per-instance stats
    all_ious: List[float] = []
    all_dices: List[float] = []

    tp_rows: List[np.ndarray] = []
    target_cls: List[int] = []
    pred_cls: List[int] = []
    conf: List[float] = []

    # Iterate through all annotations (instances) in val
    ann_ids = coco.getAnnIds()
    if cfg.max_instances is not None:
        ann_ids = ann_ids[: cfg.max_instances]

    # Cache images to avoid re-reading per instance
    image_cache: Dict[int, Tuple[np.ndarray, dict]] = {}


    vis_dir = result_path / "outputs" / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    vis_cache = {}  # image_id -> canvas


    for k, ann_id in enumerate(ann_ids):
        ann = coco.loadAnns([ann_id])[0]
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if img_id not in image_cache:
            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            img_path = img_dir / file_name
            image_rgb = load_image_rgb(img_path)
            image_cache[img_id] = (image_rgb, img_info)

        image_rgb, img_info = image_cache[img_id]
        h, w = int(img_info["height"]), int(img_info["width"])

        # GT mask
        gt_mask = coco_ann_to_binary_mask(coco, ann, h=h, w=w)

        if gt_mask.sum() == 0:
            continue

        # Prompt box: from COCO bbox (xywh) + 5% padding
        box_xyxy = xywh_to_xyxy(np.array(ann["bbox"], dtype=float))
        box_xyxy = pad_box_xyxy(box_xyxy, cfg.pad_frac, img_w=w, img_h=h)

        # SAM2 prediction (binary mask)
        pred_mask = predictor.predict_mask(image_rgb=image_rgb, box_xyxy=box_xyxy)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Pred mask shape {pred_mask.shape} != GT shape {gt_mask.shape} for ann_id={ann_id}")

        # Metrics
        iou = mask_iou(pred_mask, gt_mask)
        dice = mask_dice(pred_mask, gt_mask)
        all_ious.append(iou)
        all_dices.append(dice)

        # TP vector across IoU thresholds
        tp_vec = (iou >= cfg.iou_thrs).astype(np.uint8)  # shape (T,)
        tp_rows.append(tp_vec)

        # For your ap_per_class() integration:
        # - We assign pred_cls == target_cls (we're not evaluating classification)
        # - confidence is constant 1.0 (no ranking)
        target_cls.append(cat_id)
        pred_cls.append(cat_id)
        conf.append(1.0)

        # --- Visualization ---
        if img_id not in vis_cache:
            vis_cache[img_id] = image_rgb.copy()

        canvas = vis_cache[img_id]

        # Colors (BGR for OpenCV)
        GT_COLOR = (0, 255, 0)     # green
        PRED_COLOR = (0, 0, 255)   # red
        BOX_COLOR = (255, 0, 0)    # blue

        canvas = overlay_mask(canvas, gt_mask, GT_COLOR, alpha=0.35)
        canvas = overlay_mask(canvas, pred_mask, PRED_COLOR, alpha=0.35)
        canvas = draw_box(canvas, box_xyxy, BOX_COLOR)

        label = f"{names.get(cat_id, cat_id)} | IoU {iou:.2f} | Dice {dice:.2f}"
        canvas = put_text(canvas, label, (int(box_xyxy[0]), max(15, int(box_xyxy[1]) - 5)))

        vis_cache[img_id] = canvas


    tp = np.stack(tp_rows, axis=0) if tp_rows else np.zeros((0, len(cfg.iou_thrs)), dtype=np.uint8)
    conf = np.array(conf, dtype=float)
    pred_cls = np.array(pred_cls, dtype=int)
    target_cls = np.array(target_cls, dtype=int)

    # Summaries
    results = {
        "tp": tp,
        "conf": conf,
        "pred_cls": pred_cls,
        "target_cls": target_cls,
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


# -----------------------------
# Example: run + integrate with your ap_per_class()
# -----------------------------


import cv2
import numpy as np
from pathlib import Path


def overlay_mask(image, mask, color, alpha=0.5):
    """
    image: (H, W, 3) uint8
    mask: (H, W) uint8 {0,1}
    color: (B, G, R)
    """
    overlay = image.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def draw_box(image, box_xyxy, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2 = box_xyxy.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def put_text(image, text, org, color=(255, 255, 255)):
    return cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

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
    # dataset_root = Path("/u/slee32/data/vinay_ent/JS_R_Tonsil_coco_merged_v3_vv4")  # adjust if needed
    dataset_root = Path("/projects/illinois/cimed/bts/gayed/data/processed_data/coco_format/1200images")
    result_path = Path("/projects/illinois/cimed/bts/gayed/data/train_script/SAM2_project/sam2/result/sam2/"+dataset_root.stem)
    result_path = get_unique_path(result_path)


    cfg = EvalConfig(dataset_root=dataset_root, pad_frac=0.05, max_instances=None)

    predictor = SAM21Predictor(
        checkpoint_path="/u/sl257/data/checkpoint/sam2.1_hiera_large.pt",
        device="cuda",
    )

    out = evaluate_sam2_on_coco_val(cfg, predictor, result_path)

    print("Instances:", out["num_instances"])
    print("Mean IoU:", out["mean_iou"])
    print("Mean Dice:", out["mean_dice"])

    # --- Connect to YOUR ap_per_class() ---
    from yolo_metrics import ap_per_class
    
    # names: your function expects dict[int,str] where keys are class ids used in target_cls/pred_cls.
    # We already have category_id -> name
    
    from pathlib import Path

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

        ap50 = ap[i, 0]                 # AP@0.5
        ap5095 = ap[i].mean()           # AP@0.5:0.95

        print(f"{cls_name:20s} | AP@0.5: {ap50:.4f} | AP@0.5:0.95: {ap5095:.4f}")

    per_class_ap = {}

    for i, cls_id in enumerate(unique_classes):
        cls_name = names.get(cls_id, str(cls_id))
        per_class_ap[cls_name] = {
            "AP@0.5": float(ap[i, 0]),
            "AP@0.5:0.95": float(ap[i].mean()),
        }

    with open(result_path / "per_class_ap.json", "w") as f:
        json.dump(per_class_ap, f, indent=2)
    

    from collections import defaultdict

    iou_by_class = defaultdict(list)
    dice_by_class = defaultdict(list)

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

    # ap has shape (nc, num_iou_thresholds). Common summaries:
    # mAP@0.5 is ap[:,0].mean()
    # mAP@0.5:0.95 is ap.mean()
    # per-class AP@0.5 is ap[:,0]
    
    print("mAP@0.5:", ap[:, 0].mean())
    print("mAP@0.5:0.95:", ap.mean())
