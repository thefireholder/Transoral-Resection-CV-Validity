"""
plot_config.py -- every knob for the figures lives here. Edit this, then
re-run  python plot/make_figures.py  (nothing else needs touching).
"""
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1]
DATA = RESULT / "data"
FIG_DIR = RESULT / "figures"

# ---------------------------------------------------------------------------
# Models: key = name used in data/ files; order here = order in every figure.
# Comment a line out to drop a model from all figures; add "sam2" to include SAM2
# (its numbers are not collected yet -- see README "SAM2").
# ---------------------------------------------------------------------------
# Every model key that can appear anywhere. label = legend / panel title, short = table header,
# color = Okabe-Ito (colour-blind safe).
ALL_MODELS = {
    "maskrcnn":     {"label": "Mask R-CNN",               "short": "Mask R-CNN",  "color": "#0072B2"},
    "yolo":         {"label": "YOLO11n-seg",              "short": "YOLO11n",     "color": "#E69F00"},
    "sam3_yolobox": {"label": "SAM3 (YOLO boxes)",        "short": "SAM3+YOLO",   "color": "#009E73"},
    "sam3_text":    {"label": "SAM3 (text prompt, zero-shot)", "short": "SAM3 text", "color": "#56B4E9"},
    "sam3_gtbox":   {"label": "SAM3 (GT boxes, oracle)",  "short": "SAM3 GTbox", "color": "#999999"},
    "monai":        {"label": "MONAI UNet",               "short": "MONAI",       "color": "#CC79A7"},
}
# The MAIN comparison (Figs 1b, 2, 3, 4): one column per system. SAM3 enters as the two-stage
# YOLO-boxes -> SAM3 pipeline (decision 2026-09-18: the zero-shot text-prompted SAM3 scores ~0.03 mAP on
# this vocabulary and would only raise questions in the main figures; it and the GT-box oracle are shown
# in the ablation Figs 3b/3c and explained in the manuscript).
MAIN_MODELS = ["maskrcnn", "yolo", "sam3_yolobox", "monai"]
MODELS = {k: ALL_MODELS[k] for k in MAIN_MODELS}

# Figure 3b -- SAM3 prompting ablation table: rows in this order, then the reference rows.
FIG3B_ROWS = ["sam3_yolobox", "sam3_gtbox", "sam3_text"]   # main-figure condition first
FIG3B_REFERENCE = ["yolo"]                    # shown below a rule, e.g. YOLO alone vs YOLO boxes + SAM3
FIG3B_METRICS = ["AP50", "AP50_95"]            # per-class columns, like Fig 3
FIG3B_EXTRA_AGG = []                           # extra aggregate rows under the table, e.g. [("iou", "mean IoU")]
# Figure 3c -- compact aggregate table of the same rows: six metrics + a "what it tests" note per row.
FIG3C_METRICS = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("precision", "P"), ("recall", "R"),
                 ("iou", "IoU"), ("dice", "Dice")]
FIG3B_SHOW_NOTES = False                      # notes below: kept for the manuscript caption, not drawn by default
FIG3B_NOTES = {                               # one line per row (drawn only if FIG3B_SHOW_NOTES)
    "sam3_text":    "class-name text prompts: detect + classify + segment, zero-shot",
    "sam3_yolobox": "YOLO11n boxes (+class, +score) as prompts: SAM3 only draws the mask",
    "sam3_gtbox":   "GT boxes as prompts: perfect localisation + class = upper bound",
    "yolo":         "reference: the detector whose boxes feed the row above",
}

# Which numbers feed the table / bar chart:
#   "reported"   = each model's own original evaluation (mixed AP conventions, see README)
#   "recomputed" = scripts/compute_mask_metrics.py, one rule for all (n/p where no prediction file)
METRIC_SOURCE = "recomputed"
# If a value is n/p in METRIC_SOURCE but exists in the other source (e.g. YOLO Dice/IoU only exist as
# recomputed), use it -- marked with "*" in the table and hatched in the bar chart. False = leave n/p.
FILL_FROM_OTHER_SOURCE = True
SPLIT = "test"

# Class order used in every per-class figure/table (14 classes present in the test split).
CLASSES = [
    "base of tongue", "bipolar", "bot", "cut", "endotracheal tube", "ligasure",
    "maryland", "medial pterygoid muscle", "monopolar", "parapharyngeal fat",
    "posterior pharyngeal wall", "prevertebral fascia", "soft palate", "suction",
]
# Fixed colour per class (tab20 order), shared by PR curves and mask overlays.
import matplotlib
_tab20 = matplotlib.colormaps["tab20"].colors
CLASS_COLORS = {c: _tab20[i % 20] for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Figure 1a  training curves
# ---------------------------------------------------------------------------
FIG1A_ROWS = [  # (column in data/training_curves/<model>.csv pairs, y label)
    (("train_loss", "val_loss"), "loss"),
    (("train_precision", "val_precision"), "mask precision"),
    (("train_recall", "val_recall"), "mask recall"),
]
FIG1A_SPLIT_STYLE = {"train": dict(linestyle="-"), "val": dict(linestyle="--")}
FIG1A_MODELS = ["maskrcnn", "yolo", "monai"]  # SAM3 variants are not trained

# ---------------------------------------------------------------------------
# Figure 1b  qualitative overlays  (GT | model 1 | model 2 | ...)
# ---------------------------------------------------------------------------
FIG1B_IMAGES = [                              # test-split file names
    "JML619-images__frame_35094.jpg",         # 8 classes, JM L tonsil video
    "201-400-image__frame_00400.jpg",         # 6 classes, JS R tonsil video
]
FIG1B_SCORE_THRESHOLD = {"maskrcnn": 0.5, "yolo": 0.25, "sam3_text": 0.5, "sam3_yolobox": 0.25,
                         "sam3_gtbox": 0.0, "monai": 0.0}
FIG1B_ALPHA = 0.45
FIG1B_DRAW_CONTOUR = True
FIG1B_CROP = None                             # e.g. (x0, y0, x1, y1) in pixels to zoom, or None
# Models without a prediction file can show a pre-rendered image instead (clearly labelled "own frame"):
# {model: [image per row]}; "right_half" takes the prediction side of a GT|pred side-by-side jpg.
_MONAI_DIR = RESULT / "some disorganized results" / "1200 images" / "monai-results"
FIG1B_FALLBACK = {}   # off: MONAI's jpgs are different frames/colours and read badly next to the GT column.
# FIG1B_FALLBACK = {"monai": [_MONAI_DIR / "qualitative_median.jpg", _MONAI_DIR / "qualitative_best.jpg"]}
FIG1B_FALLBACK_CROP = "right_half"            # "right_half" | "left_half" | None

# ---------------------------------------------------------------------------
# Figure 2  PR curves (mask, IoU 0.5)
# ---------------------------------------------------------------------------
FIG2_LAYOUTS = ["per_model", "per_class"]   # both are always rendered:
                              # "per_model": one panel per model, a line per CLASS (class colours, like MaskPR_curve.png)
                              # "per_class": one panel per class, a line per MODEL (model colours)
FIG2_SHOW_MEAN = True

# ---------------------------------------------------------------------------
# Figure 3  mAP table
# ---------------------------------------------------------------------------
FIG3_METRICS = ["AP50", "AP50_95"]           # any of AP50, AP50_95, precision, recall, iou, dice
FIG3_DECIMALS = 3
FIG3_ALSO_WRITE = ["csv", "md", "tex"]       # text versions next to the png

# ---------------------------------------------------------------------------
# Figure 4  aggregate bar chart
# ---------------------------------------------------------------------------
FIG4_METRICS = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("dice", "Dice"), ("iou", "IoU")]
FIG4_ANNOTATE = True

# ---------------------------------------------------------------------------
# Figure 5  dataset-size sweep  (reads data/dataset_size_sweep.csv)
# ---------------------------------------------------------------------------
FIG5_METRIC = "mAP50"                        # mAP50 | mAP50_95 | dice | iou
FIG5_SIZES = [75, 150, 300, 600, 900, 1200]

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
DPI = 200
FORMATS = ["png", "pdf"]
RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "figure.facecolor": "white", "savefig.bbox": "tight",
}
NP_TEXT = "n/p"     # what an empty panel / cell says
