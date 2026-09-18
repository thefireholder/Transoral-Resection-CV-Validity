#!/usr/bin/env python3
"""
collect_results.py -- gather every number that already exists on disk into
ONE place (result/data/) with one schema for all models, and write
data/SOURCES.md saying where each number came from.

CPU only, seconds. Safe to re-run at any time (overwrites data/metrics_*.csv,
data/training_curves/*.csv and data/SOURCES.md -- so do NOT hand-edit those;
edit this script or the source files instead).

Outputs
  data/metrics_per_class.csv   model,split,source,class,AP50,AP50_95,precision,recall,iou,dice
  data/metrics_aggregate.csv   model,split,source,mAP50,mAP50_95,precision,recall,iou,dice,n_classes
  data/training_curves/<model>.csv
        epoch,train_loss,val_loss,train_precision,train_recall,
        val_precision,val_recall,val_mAP50,val_mAP50_95         (mask metrics)
        (if data/training_curves/<model>_rerun.csv exists -- written by
         scripts/rerun/*_train_curves.py -- it is copied over <model>.csv instead)
  data/SOURCES.md

`source` column:
  reported    = the model's own original evaluation code (numbers as they
                appear in the run directories / gathered folder)
  recomputed  = scripts/compute_mask_metrics.py on the uniform prediction
                files (same matching + COCO AP rule for every model)
Missing values are written as  n/p  (not produced).

All metrics are MASK metrics (never bounding box).
"""
import csv
import json
import re
import sys
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1]
DATA = RESULT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seglib import SHARED  # noqa: E402  (honours $TORS_SHARED)
GATHERED = RESULT / "some disorganized results" / "1200 images"

MRCNN = SHARED / "train_script/maskrcnn_project"
YOLO = SHARED / "train_script/yolov12_project/runs/segment/1200images"
SAM = SHARED / "train_script/SAM2_project/sam2"

NP = "n/p"
MODELS = ["maskrcnn", "yolo", "sam3_text", "sam3_yolobox", "sam3_gtbox", "monai"]
SPLIT = "test"
# The 14 classes that have GT instances in the test split, alphabetical.
TEST_CLASSES = [
    "base of tongue", "bipolar", "bot", "cut", "endotracheal tube", "ligasure",
    "maryland", "medial pterygoid muscle", "monopolar", "parapharyngeal fat",
    "posterior pharyngeal wall", "prevertebral fascia", "soft palate", "suction",
]
PER_CLASS_COLS = ["AP50", "AP50_95", "precision", "recall", "iou", "dice"]
AGG_COLS = ["mAP50", "mAP50_95", "precision", "recall", "iou", "dice"]
CURVE_COLS = ["epoch", "train_loss", "val_loss", "train_precision", "train_recall",
              "val_precision", "val_recall", "val_mAP50", "val_mAP50_95"]

sources = []   # (model, what, path, note)


def src(model, what, path, note=""):
    sources.append((model, what, str(path), note))


def fmt(x):
    return NP if x is None else f"{float(x):.6f}"


# ---------------------------------------------------------------------------
# Mask R-CNN
# ---------------------------------------------------------------------------
def maskrcnn():
    p = MRCNN / "run_output/mask_metrics/mask_metrics_summary.json"
    txt = GATHERED / "rcnn-1200/rcnn_result.txt"
    if not p.exists() and not txt.exists():
        return _missing("maskrcnn", [p, txt])
    adopted = DATA / "predictions" / f"maskrcnn_{SPLIT}.json"
    if adopted.exists() and json.load(open(adopted)).get("model") == "maskrcnn_rerun":
        # the 12-epoch rerun was adopted (--adopt): the old checkpoint's numbers no longer describe
        # the model in the figures -> reported rows become n/p, use source=recomputed.
        src("maskrcnn", "reported per-class / aggregate metrics", "-",
            "n/p: SUPERSEDED. data/predictions/maskrcnn_test.json now holds the re-trained model "
            "(scripts/rerun/maskrcnn_train_curves.py --adopt, see scripts/rerun/output/maskrcnn/recipe.txt); "
            "its metrics are the source=recomputed rows. The old 5-epoch (am232) numbers remain in "
            f"{p}.")
        per = {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}
        agg = {k: None for k in AGG_COLS}
        _maskrcnn_curve()
        return per, agg
    if p.exists():
        d = json.load(open(p))[SPLIT]
    else:                                   # train_script/ deleted 2026-09-17: parse the gathered text report instead
        d = _parse_rcnn_result_txt(txt, SPLIT)
        p = txt
    src("maskrcnn", "per-class + aggregate AP50, AP50:95, IoU, Dice (test)", p,
        "produced by eval_saved_mask_metrics.py (COCOeval segm, 101-pt AP; IoU/Dice = mean over "
        "greedy IoU>=0.5 matched instances, score>=0.05). Same numbers as gathered "
        "rcnn-1200/rcnn_result.txt.")
    per = {}
    for cls in TEST_CLASSES:
        c = d["per_class"].get(cls, {})
        per[cls] = {"AP50": c.get("mask_ap_0.5"), "AP50_95": c.get("mask_ap_0.5_0.95"),
                    "precision": None, "recall": None,
                    "iou": c.get("mean_mask_iou"), "dice": c.get("mean_mask_dice")}
    a = d["aggregate"]
    agg = {"mAP50": a["mask_mAP_0.5"], "mAP50_95": a["mask_mAP_0.5_0.95"],
           "precision": None, "recall": None, "iou": a["mean_mask_iou"], "dice": a["mean_mask_dice"]}
    src("maskrcnn", "per-class precision / recall", "-",
        "n/p: only box-based confidence curves exist (run_output/curves/*.csv, "
        "generate_curves.py matches boxes not masks). Mask-based values -> source=recomputed.")

    _maskrcnn_curve()
    return per, agg


def _parse_rcnn_result_txt(path, split):
    """some disorganized results/.../rcnn_result.txt -> same dict shape as mask_metrics_summary.json[split]."""
    text = open(path, errors="replace").read()
    block = text.split("TEST" if split == "test" else "VALIDATION")[1].split("\nTEST")[0] if split != "test" else text.split("\nTEST")[1]
    def num(v):
        v = v.strip(); return None if v in ("N/A", "") else float(v)
    per = {}
    for m in re.finditer(r"^(.+?)\s*\| AP@0\.5: ([\d.]+|N/A)\s*\| AP@0\.5:0\.95: ([\d.]+|N/A)", block, re.M):
        per.setdefault(m.group(1).strip(), {})
        per[m.group(1).strip()].update({"mask_ap_0.5": num(m.group(2)), "mask_ap_0.5_0.95": num(m.group(3))})
    for m in re.finditer(r"^(.+?)\s*\| Mean IoU: ([\d.]+|N/A)\s*\| Mean Dice: ([\d.]+|N/A)", block, re.M):
        per.setdefault(m.group(1).strip(), {})
        per[m.group(1).strip()].update({"mean_mask_iou": num(m.group(2)), "mean_mask_dice": num(m.group(3))})
    agg = {"mask_mAP_0.5": float(re.search(r"mAP@0.5: ([\d.]+)", block).group(1)),
           "mask_mAP_0.5_0.95": float(re.search(r"mAP@0.5:0.95: ([\d.]+)", block).group(1)),
           "mean_mask_iou": float(re.search(r"^Mean IoU: ([\d.]+)", block, re.M).group(1)),
           "mean_mask_dice": float(re.search(r"^Mean Dice: ([\d.]+)", block, re.M).group(1))}
    return {"per_class": per, "aggregate": agg}


def _maskrcnn_curve():
    # training curve: slurm log of the run that produced the current best checkpoint
    log = MRCNN / "slurm_9815937_full.out"
    transcribed = DATA / "training_curves" / "maskrcnn_original5ep.csv"
    if not log.exists() and transcribed.exists():   # log deleted 2026-09-17; values were transcribed from it
        src("maskrcnn", "training curve (original 5-epoch run)", transcribed,
            "per-epoch train loss + COCO val mask AP transcribed from the deleted slurm_9815937_full.out. "
            "Only used if maskrcnn_rerun.csv is absent.")
        rerun = DATA / "training_curves" / "maskrcnn_rerun.csv"
        (DATA / "training_curves" / "maskrcnn.csv").write_text((rerun if rerun.exists() else transcribed).read_text())
        return
    src("maskrcnn", "training curve (train loss, val mask AP per epoch)", log,
        "5 epochs, seed 42. Only train loss + COCO val AP/AR are logged per epoch; "
        "val loss, train-split P/R and val P/R at a confidence -> n/p (COCO AR@100 is not the same quantity). Rerun scripts/rerun/maskrcnn_train_curves.py "
        "to fill. (slurm_9844298_full.out is an earlier near-identical run.)")
    rows, epoch, seg = [], None, False
    cur = {}
    if not log.exists():          # e.g. on a laptop without train_script/ -> only the rerun csv can fill it
        write_curve("maskrcnn", rows)
        return
    for line in open(log):
        m = re.match(r"Epoch (\d+)/\d+ - train loss: ([\d.]+)", line)
        if m:
            if cur:
                rows.append(cur)
            epoch = int(m.group(1))
            cur = {"epoch": epoch, "train_loss": float(m.group(2))}
            continue
        if "Evaluate annotation type *segm*" in line:
            seg = True
        elif "Evaluate annotation type *bbox*" in line:
            seg = False
        if seg and "Average Precision" in line and "IoU=0.50 " in line:
            cur["val_mAP50"] = float(line.split("=")[-1])
        if seg and "Average Precision" in line and "IoU=0.50:0.95 | area=   all" in line:
            cur["val_mAP50_95"] = float(line.split("=")[-1])
        if line.startswith("Evaluating on test set"):
            break
    if cur:
        rows.append(cur)
    write_curve("maskrcnn", rows)


# ---------------------------------------------------------------------------
# YOLO11n-seg
# ---------------------------------------------------------------------------
def _missing(model, files):
    """On a machine without train_script/ (laptop): everything 'reported' is n/p."""
    src(model, "reported metrics + training curve", "-",
        "n/p on this machine: original run files not found (" + ", ".join(str(f) for f in files) + "). "
        "Set $TORS_SHARED or run collect_results.py on the cluster.")
    write_curve(model, [])
    return {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}, {k: None for k in AGG_COLS}


def yolo():
    log = YOLO / "yolo11n_1200images_9808510.out"
    csvp = YOLO / "1200images_test/results.csv"
    if not log.exists():            # train_script/ was deleted 2026-09-17 -> gathered copy of the test log
        log = GATHERED / "yolo-1200/test result/result.txt"
    if not log.exists():
        return _missing("yolo", [log])
    src("yolo", "per-class Mask AP50, AP50:95 (test)", log,
        "stdout of model.val(split='test') -- the per-class Mask mAP columns are ONLY in the log; "
        "1200images_test/results.csv has box mAP only. ultralytics AP rule (linear ramp to (1,0)).")
    src("yolo", "per-class Mask precision / recall (test)", csvp,
        "Mask-P / Mask-R columns; ultralytics reports P/R at the F1-maximising confidence.")
    src("yolo", "aggregate (test)", log, "'=== Test metrics ===' block: metrics/*(M). "
        "Same as gathered yolo-1200/test result/result.txt.")
    src("yolo", "per-class IoU / Dice", "-", "n/p from ultralytics; -> source=recomputed "
        "(scripts/export_yolo_predictions.py + compute_mask_metrics.py).")

    text = open(log, errors="replace").read()
    text = text.replace("\r", "\n")
    # test block = the last per-class table (after the last 'all  158  984' line)
    i = text.rfind("\n                   all        158")
    block = text[i:].splitlines()[1:]
    per = {}
    for line in block:
        m = re.match(r"\s*(.+?)\s+(\d+)\s+(\d+)\s+" + r"([\d.e-]+)\s+" * 7 + r"([\d.e-]+)\s*$", line)
        if not m:
            if per:
                break
            continue
        name = m.group(1).strip()
        vals = [float(v) for v in m.groups()[3:]]   # BoxP BoxR BoxmAP50 BoxmAP5095 MaskP MaskR MaskmAP50 MaskmAP5095
        per[name] = {"AP50": vals[6], "AP50_95": vals[7], "precision": vals[4], "recall": vals[5],
                     "iou": None, "dice": None}
    agg = {}
    for k, key in [("precision(M)", "precision"), ("recall(M)", "recall"),
                   ("mAP50(M)", "mAP50"), ("mAP50-95(M)", "mAP50_95")]:
        agg[key] = float(re.search(rf"metrics/{re.escape(k)}: ([\d.]+)", text).group(1))
    agg["iou"] = agg["dice"] = None
    per = {c: per.get(c, {k: None for k in PER_CLASS_COLS}) for c in TEST_CLASSES}

    # training curve
    tr = YOLO / "1200images_train/results.csv"
    if not tr.exists():
        tr = GATHERED / "yolo-1200/val result/results.csv"     # gathered copy of the same file
    src("yolo", "training curve", tr, "ultralytics per-epoch log, 60 epochs, seed 0, deterministic. "
        "Validation-split mask P/R/mAP per epoch; train-split P/R is never evaluated by ultralytics "
        "-> n/p. Rerun scripts/rerun/yolo_train_curves.py to fill. train/val loss = box+seg+cls+dfl.")
    rows = []
    with open(tr) as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v for k, v in r.items()}
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": sum(float(r[f"train/{k}_loss"]) for k in ["box", "seg", "cls", "dfl"]),
                "val_loss": sum(float(r[f"val/{k}_loss"]) for k in ["box", "seg", "cls", "dfl"]),
                "val_precision": float(r["metrics/precision(M)"]),
                "val_recall": float(r["metrics/recall(M)"]),
                "val_mAP50": float(r["metrics/mAP50(M)"]),
                "val_mAP50_95": float(r["metrics/mAP50-95(M)"]),
            })
    write_curve("yolo", rows)
    return per, agg


# ---------------------------------------------------------------------------
# SAM3 -- three prompting conditions (Fig 3b ablation). Only the GT-box one has
# "reported" numbers (the original sam3_model.py run); the text-prompt and
# YOLO-box variants exist only as source=recomputed (scripts/rerun/sam3/).
# ---------------------------------------------------------------------------
def sam3_text():
    src("sam3_text", "reported metrics", "-", "n/p by design: this condition (zero-shot, class-name text prompts, "
        "scripts/rerun/sam3/sam3_text_prompt.py) was never evaluated by an original script; see source=recomputed.")
    src("sam3_text", "training curve", "-", "n/p: zero-shot, not trained.")
    write_curve("sam3_text", [])
    return {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}, {k: None for k in AGG_COLS}


def sam3_yolobox():
    src("sam3_yolobox", "reported metrics", "-", "n/p by design: two-stage YOLO boxes -> SAM3 masks "
        "(scripts/rerun/sam3/sam3_yolo_box_prompt.py); see source=recomputed.")
    src("sam3_yolobox", "training curve", "-", "n/p: SAM3 is not trained; the boxes come from the YOLO run.")
    write_curve("sam3_yolobox", [])
    return {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}, {k: None for k in AGG_COLS}


def sam3_gtbox():
    apj = SAM / "result/sam3/1200images/per_class_ap.json"
    log = SAM / "logs/sam3_eval_9794464.out"
    if not (apj.exists() and log.exists()):   # deleted 2026-09-17 -> gathered copies
        apj, log = GATHERED / "sam3-1200/per_class_ap.json", GATHERED / "sam3-1200/result.txt"
    if not (apj.exists() and log.exists()):
        return _missing("sam3_gtbox", [apj, log])
    src("sam3_gtbox", "per-class AP50, AP50:95 (test)", apj,
        "sam3_model.py, ultralytics-style ap_per_class with every prediction conf=1.0 "
        "(one mask per GT box prompt, pad 5%). Same file as gathered sam3-1200/per_class_ap.json.")
    src("sam3_gtbox", "per-class + mean IoU / Dice (test)", log,
        "'Per-class IoU / Dice' block; mean over ALL 983 GT instances (not just matched). "
        "Same as gathered sam3-1200/result.txt.")
    src("sam3_gtbox", "precision / recall, training curve", "-",
        "n/p: not a trained model; and with conf=1.0 for every prediction P/R collapse to a "
        "single point (precision = fraction of prompts with IoU>=0.5).")
    src("sam3_gtbox", "PR curve numbers", "-", "n/p from the original run (only PR_curve.png was saved); "
        "scripts/rerun/sam3/sam3_gt_box_prompt.py regenerates them -> source=recomputed.")
    ap = json.load(open(apj))
    text = open(log, errors="replace").read()
    per = {}
    for cls in TEST_CLASSES:
        m = re.search(rf"^{re.escape(cls)}\s*\| Mean IoU: ([\d.]+) \| Mean Dice: ([\d.]+)", text, re.M)
        per[cls] = {"AP50": ap.get(cls, {}).get("AP@0.5"), "AP50_95": ap.get(cls, {}).get("AP@0.5:0.95"),
                    "precision": None, "recall": None,
                    "iou": float(m.group(1)) if m else None, "dice": float(m.group(2)) if m else None}
    agg = {"mAP50": float(re.search(r"mAP@0.5: ([\d.]+)", text).group(1)),
           "mAP50_95": float(re.search(r"mAP@0.5:0.95: ([\d.]+)", text).group(1)),
           "precision": None, "recall": None,
           "iou": float(re.search(r"Mean IoU: ([\d.]+)", text).group(1)),
           "dice": float(re.search(r"Mean Dice: ([\d.]+)", text).group(1))}
    write_curve("sam3_gtbox", [])
    return per, agg


# ---------------------------------------------------------------------------
# MONAI UNet (results only; model + code are on another computer)
# ---------------------------------------------------------------------------
def monai():
    p = GATHERED / "monai-results/_result_map_iou_dice_per_class.csv"
    adopted = DATA / "predictions" / f"monai_{SPLIT}.json"
    if adopted.exists() and json.load(open(adopted)).get("model") == "monai_rerun":
        src("monai", "reported per-class / aggregate metrics", "-",
            "n/p: SUPERSEDED. data/predictions/monai_test.json holds the UNet re-trained on the COCO-derived masks "
            "(scripts/rerun/monai/monai_train_curves.py, recipe in scripts/rerun/output/monai/recipe.txt); its "
            "metrics are the source=recomputed rows. The old unified-vocabulary numbers remain in "
            f"{p}.")
        write_curve("monai", [])
        return {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}, {k: None for k in AGG_COLS}
    src("monai", "per-class + aggregate AP50, AP50:95, IoU, Dice, precision, recall (test)", p,
        "Semantic UNet: AP from 8-connected components of the argmax mask (COCO 101-pt AP, score = mean "
        "softmax); Dice/IoU = per-frame semantic Dice over GT-present frames (NOT instance-matched); "
        "P/R = pooled pixel precision/recall. Class vocabulary differs (no bipolar/bot, has fat fascia) "
        "-> those rows n/p / dropped. Aggregate row used: 'AGGREGATE (all 13 classes with GT)'. "
        "See also monai-results/_results_section_numbers.txt, per_class_metrics.csv, summary.csv.")
    src("monai", "training curve, PR curve, predictions", "-",
        "n/p: model, code and per-epoch logs are not on this computer.")
    per = {c: {k: None for k in PER_CLASS_COLS} for c in TEST_CLASSES}
    agg = {k: None for k in AGG_COLS}

    def f(v):
        return None if v in ("", None) else float(v)
    with open(p) as fh:
        for r in csv.DictReader(fh):
            name = r["class"]
            if name == "AGGREGATE (all 13 classes with GT)":
                agg = {"mAP50": f(r["ap50"]), "mAP50_95": f(r["ap50_95"]), "precision": f(r["precision"]),
                       "recall": f(r["recall"]), "iou": f(r["iou"]), "dice": f(r["dice"])}
            elif name in per:
                per[name] = {"AP50": f(r["ap50"]), "AP50_95": f(r["ap50_95"]), "precision": f(r["precision"]),
                             "recall": f(r["recall"]), "iou": f(r["iou"]), "dice": f(r["dice"])}
    write_curve("monai", [])
    return per, agg


# ---------------------------------------------------------------------------
# recomputed (uniform) numbers, if compute_mask_metrics.py has been run
# ---------------------------------------------------------------------------
def recomputed(model):
    p = DATA / "computed" / f"{model}_{SPLIT}.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    src(model, f"recomputed metrics (ap_method={d.get('ap_method')})", p,
        "scripts/compute_mask_metrics.py on data/predictions/ -- identical matching + AP rule for every model.")
    per = {c: {k: d["per_class"].get(c, {}).get(k) for k in PER_CLASS_COLS} for c in TEST_CLASSES}
    return per, d["aggregate"]


# ---------------------------------------------------------------------------
def write_curve(model, rows):
    out = DATA / "training_curves" / f"{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rerun = DATA / "training_curves" / f"{model}_rerun.csv"
    if rerun.exists():          # produced by scripts/rerun/*_train_curves.py -> takes precedence
        src(model, "training curve (RERUN, overrides the log-parsed one)", rerun,
            "written by scripts/rerun/; delete/rename this file to fall back to the original log.")
        out.write_text(rerun.read_text())
        return
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CURVE_COLS)
        for r in rows:
            w.writerow([r.get("epoch")] + [fmt(r.get(c)) for c in CURVE_COLS[1:]])


def main():
    per_rows, agg_rows = [], []
    for model, fn in [("maskrcnn", maskrcnn), ("yolo", yolo), ("sam3_text", sam3_text),
                      ("sam3_yolobox", sam3_yolobox), ("sam3_gtbox", sam3_gtbox), ("monai", monai)]:
        per, agg = fn()
        for cls in TEST_CLASSES:
            per_rows.append([model, SPLIT, "reported", cls] + [fmt(per[cls][k]) for k in PER_CLASS_COLS])
        n = sum(1 for c in TEST_CLASSES if per[c]["AP50"] is not None)
        agg_rows.append([model, SPLIT, "reported"] + [fmt(agg[k]) for k in AGG_COLS] + [n])
        per, agg = recomputed(model)
        if per is not None:
            for cls in TEST_CLASSES:
                per_rows.append([model, SPLIT, "recomputed", cls] + [fmt(per[cls][k]) for k in PER_CLASS_COLS])
            n = sum(1 for c in TEST_CLASSES if per[c]["AP50"] is not None)
            agg_rows.append([model, SPLIT, "recomputed"] + [fmt(agg[k]) for k in AGG_COLS] + [n])

    with open(DATA / "metrics_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "source", "class"] + PER_CLASS_COLS)
        w.writerows(per_rows)
    with open(DATA / "metrics_aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "source"] + AGG_COLS + ["n_classes"])
        w.writerows(agg_rows)

    with open(DATA / "SOURCES.md", "w") as f:
        f.write("# Where every number in data/ comes from\n\n"
                "Generated by scripts/collect_results.py -- do not edit by hand.\n"
                "`/u/sl257/shared_data` is a symlink to `/projects/illinois/cimed/bts/gayed/data`, "
                "so both spellings are the same files.\n\n")
        for model in MODELS:
            f.write(f"## {model}\n\n| what | file | note |\n|---|---|---|\n")
            for m, what, path, note in sources:
                if m == model:
                    f.write(f"| {what} | `{path}` | {note} |\n")
            f.write("\n")
    print(f"wrote {DATA/'metrics_per_class.csv'} ({len(per_rows)} rows)")
    print(f"wrote {DATA/'metrics_aggregate.csv'} ({len(agg_rows)} rows)")
    print(f"wrote {DATA/'SOURCES.md'} + data/training_curves/*.csv")


if __name__ == "__main__":
    main()
