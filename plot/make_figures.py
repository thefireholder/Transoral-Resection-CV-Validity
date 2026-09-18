#!/usr/bin/env python3
"""
make_figures.py -- draw every figure from the unified numbers in result/data/.

  conda activate pytorch_env
  python plot/make_figures.py            # all figures -> result/figures/
  python plot/make_figures.py 2 4        # only Figure 2 and Figure 4

Reads ONLY:
  data/metrics_per_class.csv, data/metrics_aggregate.csv   (Fig 3, 4)
  data/training_curves/<model>.csv                          (Fig 1a)
  data/predictions/<model>_test.json + COCO GT              (Fig 1b)
  data/pr_curves/<model>.json                               (Fig 2)
  data/dataset_size_sweep.csv                               (Fig 5)
Anything missing is drawn as an empty panel labelled  n/p  -- the code never
fails because a number is absent. All layout / colour / label choices are in
plot_config.py; each fig_N() below is independent, so edit freely.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_config as C

plt.rcParams.update(C.RC)
C.FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# data access helpers
# ---------------------------------------------------------------------------
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _other(source):
    return "recomputed" if source == "reported" else "reported"


def per_class_table(source=None, models=None):
    """{model: {class: {metric: float|nan}}} for SPLIT / METRIC_SOURCE.
    Also returns a set of (model, class, metric) that were filled from the other source."""
    source = source or C.METRIC_SOURCE
    models = list(models or C.MODELS)
    tabs = {s: {m: {} for m in models} for s in ("reported", "recomputed")}
    for r in read_csv(C.DATA / "metrics_per_class.csv"):
        if r["model"] in models and r["split"] == C.SPLIT:
            tabs[r["source"]][r["model"]][r["class"]] = {k: _num(v) for k, v in r.items()
                                                         if k not in ("model", "split", "source", "class")}
    out, filled = tabs[source], set()
    if C.FILL_FROM_OTHER_SOURCE:
        for m in models:
            for cls, vals in tabs[_other(source)][m].items():
                out[m].setdefault(cls, {})
                for k, v in vals.items():
                    if np.isnan(out[m][cls].get(k, np.nan)) and not np.isnan(v):
                        out[m][cls][k] = v; filled.add((m, cls, k))
    return out, filled


def aggregate_table(source=None, models=None):
    source = source or C.METRIC_SOURCE
    models = list(models or C.MODELS)
    tabs = {s: {} for s in ("reported", "recomputed")}
    for r in read_csv(C.DATA / "metrics_aggregate.csv"):
        if r["model"] in models and r["split"] == C.SPLIT:
            tabs[r["source"]][r["model"]] = {k: _num(v) for k, v in r.items() if k not in ("model", "split", "source")}
    out, filled = tabs[source], set()
    if C.FILL_FROM_OTHER_SOURCE:
        for m, vals in tabs[_other(source)].items():
            out.setdefault(m, {})
            for k, v in vals.items():
                if np.isnan(out[m].get(k, np.nan)) and not np.isnan(v):
                    out[m][k] = v; filled.add((m, k))
    return out, filled


def training_curve(model):
    rows = read_csv(C.DATA / "training_curves" / f"{model}.csv")
    if not rows:
        return None
    cols = rows[0].keys()
    return {c: np.array([_num(r[c]) for r in rows]) for c in cols}


def np_panel(ax, text=None):
    ax.text(0.5, 0.5, text or C.NP_TEXT, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="0.45")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def save(fig, name):
    for ext in C.FORMATS:
        fig.savefig(C.FIG_DIR / f"{name}.{ext}", dpi=C.DPI)
    plt.close(fig)
    print(f"  saved figures/{name}.{{{','.join(C.FORMATS)}}}")


# ---------------------------------------------------------------------------
# Figure 1a  training curves
# ---------------------------------------------------------------------------
def fig1a():
    models = C.FIG1A_MODELS
    nrow, ncol = len(C.FIG1A_ROWS), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.4 * nrow), squeeze=False)
    for j, m in enumerate(models):
        cur = training_curve(m)
        axes[0, j].set_title(C.MODELS[m]["label"])
        for i, ((tr_col, va_col), ylabel) in enumerate(C.FIG1A_ROWS):
            ax = axes[i, j]
            drawn = False
            if cur is not None:
                for split, col in (("train", tr_col), ("val", va_col)):
                    if col in cur and not np.all(np.isnan(cur[col])):
                        ax.plot(cur["epoch"], cur[col], color=C.MODELS[m]["color"],
                                label=split, **C.FIG1A_SPLIT_STYLE[split])
                        drawn = True
                    else:
                        ax.plot([], [], color=C.MODELS[m]["color"], label=f"{split} ({C.NP_TEXT})",
                                **C.FIG1A_SPLIT_STYLE[split])
            if not drawn:
                np_panel(ax)
            else:
                ax.legend(loc="best")
                if ylabel != "loss":
                    ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == nrow - 1:
                ax.set_xlabel("epoch")
    fig.tight_layout()
    save(fig, "fig1a_training_curves")


# ---------------------------------------------------------------------------
# Figure 1b  qualitative overlays
# ---------------------------------------------------------------------------
def _overlay(img, masks_with_cls, alpha, contour):
    import cv2
    out = img.astype(np.float32).copy()
    for m, cls in masks_with_cls:
        col = np.array(C.CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))[:3]) * 255
        out[m] = (1 - alpha) * out[m] + alpha * col
    out = out.astype(np.uint8)
    if contour:
        for m, cls in masks_with_cls:
            col = tuple(int(c * 255) for c in C.CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))[:3])
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, col, 3)
    return out


def fig1b():
    from seglib import COCO_1200, load_coco_gt, load_predictions, rle_decode
    import cv2
    gt = load_coco_gt(C.SPLIT)
    preds = {}
    for m in C.MODELS:
        p = C.DATA / "predictions" / f"{m}_{C.SPLIT}.json"
        if p.exists():
            d = load_predictions(p)
            preds[m] = {im["file_name"]: im for im in d["images"]}
    cols = ["GT"] + list(C.MODELS)
    fig, axes = plt.subplots(len(C.FIG1B_IMAGES), len(cols),
                             figsize=(3.2 * len(cols), 1.9 * len(C.FIG1B_IMAGES) + 0.6), squeeze=False)
    used_classes = set()
    for i, fn in enumerate(C.FIG1B_IMAGES):
        img = cv2.cvtColor(cv2.imread(str(COCO_1200 / "images" / C.SPLIT / fn)), cv2.COLOR_BGR2RGB)
        g = gt["images"][fn]
        for j, key in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if key == "GT":
                items = [(rle_decode(x["rle"]), x["class"]) for x in g["gts"]]
            elif key in preds and fn in preds[key]:
                thr = C.FIG1B_SCORE_THRESHOLD.get(key, 0.0)
                items = [(rle_decode(p["rle"]), p["class"]) for p in preds[key][fn]["predictions"]
                         if p["score"] >= thr]
            else:
                fb = C.FIG1B_FALLBACK.get(key, [])
                if i < len(fb) and Path(fb[i]).exists():
                    im = cv2.cvtColor(cv2.imread(str(fb[i])), cv2.COLOR_BGR2RGB)
                    if C.FIG1B_FALLBACK_CROP == "right_half":
                        im = im[:, im.shape[1] // 2:]
                    elif C.FIG1B_FALLBACK_CROP == "left_half":
                        im = im[:, : im.shape[1] // 2]
                    ax.imshow(im)
                    ax.text(0.5, 0.02, "own frame, own colours*", transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=6, color="white", bbox=dict(facecolor="black", alpha=0.5, lw=0))
                else:
                    np_panel(ax)
                if i == 0:
                    ax.set_title(C.MODELS[key]["label"])
                continue
            used_classes.update(c for _, c in items)
            ov = _overlay(img, items, C.FIG1B_ALPHA, C.FIG1B_DRAW_CONTOUR)
            if C.FIG1B_CROP:
                x0, y0, x1, y1 = C.FIG1B_CROP
                ov = ov[y0:y1, x0:x1]
            ax.imshow(ov)
            if i == 0:
                ax.set_title("Ground truth" if key == "GT" else C.MODELS[key]["label"])
        axes[i, 0].set_ylabel(fn.replace(".jpg", ""), fontsize=6)
    handles = [plt.Line2D([], [], color=C.CLASS_COLORS[c], lw=6, label=c)
               for c in C.CLASSES if c in used_classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if any(C.FIG1B_FALLBACK.get(m) for m in C.MODELS if m not in preds):
        fig.text(0.01, 0.005, "* pre-rendered by that model's own pipeline on a different test frame "
                 "(no masks available on this computer)", fontsize=6, color="0.35")
    fig.tight_layout(rect=(0, 0.12 / max(1, len(C.FIG1B_IMAGES) * 0.6), 1, 1))
    save(fig, "fig1b_qualitative")


# ---------------------------------------------------------------------------
# Figure 2  PR curves
# ---------------------------------------------------------------------------
def _pr(model):
    p = C.DATA / "pr_curves" / f"{model}.json"
    return json.load(open(p)) if p.exists() else None


def fig2():
    """Renders EVERY layout in C.FIG2_LAYOUTS so no stale file is left behind."""
    for layout in C.FIG2_LAYOUTS:
        _fig2(layout)


def _fig2(layout):
    curves = {m: _pr(m) for m in C.MODELS}
    if layout == "per_model":
        n = len(C.MODELS)
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
        for ax, m in zip(axes[0], C.MODELS):
            ax.set_title(C.MODELS[m]["label"])
            d = curves[m]
            if d is None:
                np_panel(ax); continue
            rec = np.array(d["recall"])
            ys, aps = [], []
            for cls in C.CLASSES:
                if cls not in d["classes"]:
                    continue
                y = np.array(d["classes"][cls]["precision"]); ys.append(y)
                ap = d["classes"][cls]["AP50"]
                if ap is not None:
                    aps.append(ap)
                ax.plot(rec, y, color=C.CLASS_COLORS[cls], lw=0.9, label=cls)
            if C.FIG2_SHOW_MEAN and ys:
                ax.plot(rec, np.mean(ys, 0), color="black", lw=2.2, label="all classes (mean)")
            ax.set_title(f"{C.MODELS[m]['label']}\nmAP@0.5 = {np.mean(aps):.3f}" if aps else C.MODELS[m]["label"])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("recall")
        axes[0, 0].set_ylabel("precision")
        handles = [plt.Line2D([], [], color=C.CLASS_COLORS[c], lw=1.5, label=c) for c in C.CLASSES]
        if C.FIG2_SHOW_MEAN:
            handles.append(plt.Line2D([], [], color="black", lw=2.2, label="all classes (mean)"))
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:  # per_class
        ncol = 5
        nrow = int(np.ceil(len(C.CLASSES) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)
        for k, cls in enumerate(C.CLASSES):
            ax = axes[k // ncol, k % ncol]
            ax.set_title(cls, fontsize=8)
            any_ = False
            for m in C.MODELS:
                d = curves[m]
                if d and cls in d["classes"]:
                    ax.plot(d["recall"], d["classes"][cls]["precision"], color=C.MODELS[m]["color"],
                            label=C.MODELS[m]["label"]); any_ = True
                else:
                    ax.plot([], [], color=C.MODELS[m]["color"], label=f"{C.MODELS[m]['label']} ({C.NP_TEXT})")
            if not any_:
                np_panel(ax)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        for k in range(len(C.CLASSES), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(C.MODELS), frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.supxlabel("recall", y=0.045); fig.supylabel("precision")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.suptitle(f"Mask precision-recall (IoU 0.5), {C.SPLIT} split", y=1.02)
    save(fig, f"fig2_pr_curves_{layout}")


# ---------------------------------------------------------------------------
# Figure 3  mAP table  (and the shared table builder used by Figure 3b)
# ---------------------------------------------------------------------------
METRIC_LABEL = {"AP50": "AP@0.5", "AP50_95": "AP@0.5:0.95", "precision": "P", "recall": "R", "iou": "IoU", "dice": "Dice"}
AGG_KEY = {"AP50": "mAP50", "AP50_95": "mAP50_95"}


def _per_class_ap_table(models, metrics, stem, title, extra_agg=None, footnotes=()):
    """Per-class table: one row per class + 'mean (aggregate)'; columns = model x metric.
    extra_agg: list of (aggregate metric key, label) appended as extra rows (values under every model's first
    metric column, other columns blank) -- used by Fig 3b for P/R/IoU/Dice."""
    (per, per_filled), (agg, agg_filled) = per_class_table(models=models), aggregate_table(models=models)
    header = ["class"] + [f"{C.ALL_MODELS[m]['short']}\n{METRIC_LABEL[k]}" for m in models for k in metrics]
    rows = [[cls] + [per[m].get(cls, {}).get(k, np.nan) for m in models for k in metrics] for cls in C.CLASSES]
    rows.append(["mean (aggregate)"] + [agg.get(m, {}).get(AGG_KEY.get(k, k), np.nan) for m in models for k in metrics])
    n_main = len(rows)
    for key, lab in (extra_agg or []):
        r = [lab]
        for m in models:
            r += [agg.get(m, {}).get(key, np.nan)] + [""] * (len(metrics) - 1)
        rows.append(r)

    def cell(v, star=False):
        if isinstance(v, float):
            return C.NP_TEXT if np.isnan(v) else f"{v:.{C.FIG3_DECIMALS}f}" + ("*" if star else "")
        return str(v)
    text = []
    for r_i, r in enumerate(rows):
        cls = r[0]
        stars = []
        for m in models:
            for k in metrics:
                if r_i < len(C.CLASSES):
                    stars.append((m, cls, k) in per_filled)
                elif r_i == len(C.CLASSES):
                    stars.append((m, AGG_KEY.get(k, k)) in agg_filled)
                else:
                    stars.append((m, (extra_agg or [])[r_i - n_main][0]) in agg_filled)
        text.append([cell(r[0])] + [cell(v, st) for v, st in zip(r[1:], stars)])
    any_star = any("*" in c for r in text for c in r)

    ncol = len(header)
    widths = [0.20] + [0.80 / (ncol - 1)] * (ncol - 1)
    fig, ax = plt.subplots(figsize=(2.2 + 0.95 * (ncol - 1), 0.27 * (len(rows) + 3) + 0.25 * len(footnotes)))
    ax.axis("off")
    tab = ax.table(cellText=text, colLabels=header, colWidths=widths, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1, 1.25)
    for (r, c), cellobj in tab.get_celld().items():
        cellobj.set_edgecolor("0.8")
        if r == 0 or r == n_main:
            cellobj.set_text_props(weight="bold")
        if r == 0:
            cellobj.set_facecolor("0.93"); cellobj.set_height(cellobj.get_height() * 2)
        if r > n_main:
            cellobj.set_facecolor("0.97")
        if c == 0:
            cellobj.set_text_props(ha="left"); cellobj.PAD = 0.03
    # bold best per metric column (per class row and the aggregate row)
    for r_i in range(1, n_main + 1):
        r = rows[r_i - 1]
        for k_i, k in enumerate(metrics):
            vals = [(r[1 + m_i * len(metrics) + k_i], m_i) for m_i in range(len(models))]
            vals = [(v, m_i) for v, m_i in vals if isinstance(v, float) and not np.isnan(v)]
            if vals:
                best = max(vals)[1]
                tab[r_i, 1 + best * len(metrics) + k_i].set_text_props(weight="bold")
    ax.set_title(title, fontsize=9)
    y = 0.09
    if any_star:
        fig.text(0.14, y, f"* taken from source={_other(C.METRIC_SOURCE)} (missing in {C.METRIC_SOURCE})", fontsize=6, color="0.35")
        y -= 0.035
    for line in footnotes:
        fig.text(0.14, y, line, fontsize=6, color="0.35"); y -= 0.035
    save(fig, stem)

    out = C.FIG_DIR / stem
    hdr = [h.replace("\n", " ") for h in header]
    if "csv" in C.FIG3_ALSO_WRITE:
        with open(out.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(text)
    if "md" in C.FIG3_ALSO_WRITE:
        with open(out.with_suffix(".md"), "w") as f:
            f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for r in text:
                f.write("| " + " | ".join(r) + " |\n")
    if "tex" in C.FIG3_ALSO_WRITE:
        with open(out.with_suffix(".tex"), "w") as f:
            f.write("\\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}\n\\toprule\n")
            f.write(" & ".join(hdr) + " \\\\\n\\midrule\n")
            for r in text[:n_main - 1]:
                f.write(" & ".join(r) + " \\\\\n")
            f.write("\\midrule\n")
            for r in text[n_main - 1:]:
                f.write(" & ".join(r) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")


def fig3():
    _per_class_ap_table(list(C.MODELS), C.FIG3_METRICS, "fig3_map_table",
                        f"Mask AP per class, {C.SPLIT} split  (source: {C.METRIC_SOURCE})")


# ---------------------------------------------------------------------------
# Figure 3b  SAM3 prompting ablation: per-class AP for each prompt condition (+ YOLO reference)
# ---------------------------------------------------------------------------
def fig3b():
    models = list(C.FIG3B_ROWS) + list(C.FIG3B_REFERENCE)
    notes = [f"{C.ALL_MODELS[k]['short']}: {C.FIG3B_NOTES[k]}" for k in models if k in C.FIG3B_NOTES] \
        if C.FIG3B_SHOW_NOTES else []
    _per_class_ap_table(models, C.FIG3B_METRICS, "fig3b_sam3_prompt_ablation_per_class",
                        f"SAM3 prompting ablation -- mask AP per class, {C.SPLIT} split  (source: {C.METRIC_SOURCE})",
                        extra_agg=C.FIG3B_EXTRA_AGG, footnotes=notes)


# ---------------------------------------------------------------------------
# Figure 3c  SAM3 prompting ablation: aggregate metrics per prompt condition, one row each
# ---------------------------------------------------------------------------
def fig3c():
    rows_keys = list(C.FIG3B_ROWS) + list(C.FIG3B_REFERENCE)
    agg, filled = aggregate_table(models=rows_keys)
    header = ["SAM3 prompt condition"] + [lab for _, lab in C.FIG3C_METRICS] + (["what it tests"] if C.FIG3B_SHOW_NOTES else [])
    text = []
    for k in rows_keys:
        vals = []
        for m, _ in C.FIG3C_METRICS:
            v = agg.get(k, {}).get(m, np.nan)
            vals.append(C.NP_TEXT if np.isnan(v) else f"{v:.3f}" + ("*" if (k, m) in filled else ""))
        text.append([C.ALL_MODELS[k]["label"]] + vals + ([C.FIG3B_NOTES.get(k, "")] if C.FIG3B_SHOW_NOTES else []))
    ncol = len(header)
    widths = ([0.16] + [0.08] * len(C.FIG3C_METRICS) + [0.36]) if C.FIG3B_SHOW_NOTES else \
             ([0.25] + [0.125] * len(C.FIG3C_METRICS))
    fig, ax = plt.subplots(figsize=(14 if C.FIG3B_SHOW_NOTES else 10, 0.45 * (len(text) + 2)))
    ax.axis("off")
    tab = ax.table(cellText=text, colLabels=header, colWidths=widths, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("0.93")
        if c == 0 or (C.FIG3B_SHOW_NOTES and c == ncol - 1):
            cell.set_text_props(ha="left"); cell.PAD = 0.02
        if r == len(C.FIG3B_ROWS) + 1 and C.FIG3B_REFERENCE:      # rule above the reference rows
            cell.set_linewidth(1.2)
    ax.set_title(f"SAM3 prompting ablation -- aggregate, {C.SPLIT} split  (source: {C.METRIC_SOURCE})", fontsize=9)
    save(fig, "fig3c_sam3_prompt_ablation_aggregate")
    stem = C.FIG_DIR / "fig3c_sam3_prompt_ablation_aggregate"
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(text)
    with open(stem.with_suffix(".md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * ncol + "\n")
        for r in text:
            f.write("| " + " | ".join(r) + " |\n")


# ---------------------------------------------------------------------------
# Figure 4  aggregate bar chart
# ---------------------------------------------------------------------------
def fig4():
    agg, filled = aggregate_table()
    n_m, n_k = len(C.MODELS), len(C.FIG4_METRICS)
    width = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(1.8 * n_k + 1.5, 3.2))
    x = np.arange(n_k)
    for i, m in enumerate(C.MODELS):
        vals = [agg.get(m, {}).get(k, np.nan) for k, _ in C.FIG4_METRICS]
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, np.nan_to_num(vals), width * 0.92, color=C.MODELS[m]["color"], label=C.MODELS[m]["label"])
        for b, (k, _) in zip(bars, C.FIG4_METRICS):
            if (m, k) in filled:
                b.set_hatch("///"); b.set_edgecolor("white"); b.set_linewidth(0)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, C.NP_TEXT, ha="center", va="bottom",
                        rotation=90, fontsize=6, color="0.4")
            elif C.FIG4_ANNOTATE:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in C.FIG4_METRICS])
    ax.set_ylim(0, 1.08); ax.set_ylabel("score"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(C.MODELS), loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title(f"Aggregate mask metrics, {C.SPLIT} split  (source: {C.METRIC_SOURCE})")
    if filled:
        ax.text(1.0, -0.22, f"hatched = from source={_other(C.METRIC_SOURCE)}", transform=ax.transAxes,
                ha="right", fontsize=6, color="0.35")
    save(fig, "fig4_aggregate_bars")


# ---------------------------------------------------------------------------
# Figure 5  dataset-size sweep
# ---------------------------------------------------------------------------
def fig5():
    rows = read_csv(C.DATA / "dataset_size_sweep.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    drawn = False
    for m in C.MODELS:
        pts = {}
        for r in rows:
            if r["model"] == m and r["split"] == C.SPLIT and r[C.FIG5_METRIC] not in ("", C.NP_TEXT):
                pts.setdefault(int(r["n_train"]), []).append(float(r[C.FIG5_METRIC]))
        if not pts:
            ax.plot([], [], marker="o", color=C.MODELS[m]["color"], label=f"{C.MODELS[m]['label']} ({C.NP_TEXT})")
            continue
        xs = sorted(pts); ys = [np.mean(pts[k]) for k in xs]; es = [np.std(pts[k]) for k in xs]
        ax.errorbar(xs, ys, yerr=es if any(es) else None, marker="o", ms=4, capsize=2,
                    color=C.MODELS[m]["color"], label=C.MODELS[m]["label"])
        drawn = True
    if not drawn:
        np_panel(ax, f"{C.NP_TEXT}\n(data/dataset_size_sweep.csv empty -- see scripts/rerun/dataset_size_sweep/)")
    ax.set_xscale("log"); ax.set_xticks(C.FIG5_SIZES); ax.set_xticklabels([str(s) for s in C.FIG5_SIZES])
    ax.set_xlabel("# training images"); ax.set_ylabel({"mAP50": "mask mAP@0.5", "mAP50_95": "mask mAP@0.5:0.95",
                                                      "dice": "Dice", "iou": "IoU"}[C.FIG5_METRIC])
    ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.set_title("Effect of training-set size")
    save(fig, "fig5_dataset_size")


FIGS = {"1a": fig1a, "1b": fig1b, "2": fig2, "3": fig3, "3b": fig3b, "3c": fig3c, "4": fig4, "5": fig5}


if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        for key, fn in FIGS.items():
            if key == k or (k not in FIGS and key.startswith(k)):
                print(f"Figure {key}")
                fn()
