#!/usr/bin/env python3
"""
Re-train Mask R-CNN (recipe of scripts/original/maskrcnn/train_maskrcnn.py,
seed 42) while logging, EVERY EPOCH, mask precision / recall / AP on the
validation split and on a fixed seeded subset of the training split.

Schedule: the original run used 5 epochs with the LR cut x0.1 after epoch 3,
which stops learning while val AP is still rising. Default here is the
recommended longer schedule: 12 epochs, LR x0.1 after epochs 8 and 11
(--epochs 12 --lr-steps 8 11). Pass --epochs 5 --lr-steps 3 for the original.

Writes data/training_curves/maskrcnn_rerun.csv; collect_results.py then
prefers it over the log-parsed maskrcnn.csv (Figure 1a).
Also writes the uniform prediction file for the best checkpoint (by val mask
mAP50-95) on the test split:
  --adopt      -> data/predictions/maskrcnn_test.json   (REPLACES the export of
                  the old 5-epoch checkpoint; the figures then use this model.
                  collect_results.py notices and marks the old "reported"
                  Mask R-CNN numbers as superseded.)
  default      -> data/predictions/maskrcnn_rerun_test.json (kept on the side)
Re-run  scripts/export_maskrcnn_predictions.py  at any time to get the old
checkpoint's file back.

Needs a GPU (~1.5 h on V100 for 5 epochs + per-epoch eval).
Submit with:  sbatch scripts/rerun/maskrcnn_train_curves.sbatch

Dataset: processed_data/coco_format/1200images (the original run used
coco_format/maskrcnn, a copy owned by am232 that sl257 cannot read; both
hold the same 732/157/158 split and 21 categories).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[0]))                     # scripts/
sys.path.insert(0, str(HERE.parents[0] / "original" / "maskrcnn"))
from seglib import COCO_1200, evaluate_predictions, load_coco_gt, rle_encode, save_predictions  # noqa: E402
from train_maskrcnn import SurgicalMaskRCNNDataset, collate_fn, get_model, set_seed, train_one_epoch  # noqa: E402

CURVE_COLS = ["epoch", "train_loss", "val_loss", "train_precision", "train_recall",
              "val_precision", "val_recall", "val_mAP50", "val_mAP50_95"]


@torch.no_grad()
def predict_split(model, dataset, device, score_thresh, idx_to_name, indices=None):
    model.eval()
    images = []
    indices = range(len(dataset)) if indices is None else indices
    for i in indices:
        img, target = dataset[i]
        out = model([img.to(device)])[0]
        info = dataset.coco.loadImgs(int(target["image_id"]))[0]
        preds = []
        for score, label, mask in zip(out["scores"], out["labels"], out["masks"]):
            s = float(score)
            if s < score_thresh:
                continue
            m = (mask[0].cpu().numpy() > 0.5)
            if m.any():
                preds.append({"class": idx_to_name[int(label)], "score": s, "rle": rle_encode(m)})
        images.append({"file_name": info["file_name"], "height": info["height"],
                       "width": info["width"], "predictions": preds})
    return images


@torch.no_grad()
def loss_on_split(model, loader, device):
    """torchvision detection models only return losses in train mode."""
    model.train()
    tot, n = 0.0, 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        tot += float(sum(loss_dict.values())); n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=COCO_1200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr-steps", type=int, nargs="*", default=[8, 11],
                    help="epochs after which LR is multiplied by 0.1 (original run: 3)")
    ap.add_argument("--adopt", action="store_true", help="write test predictions as maskrcnn_test.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--train-eval-size", type=int, default=157,
                    help="how many (seeded, fixed) train images to evaluate per epoch; 0 = all 732")
    ap.add_argument("--output-dir", type=Path, default=HERE / "output" / "maskrcnn")
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    ds = {s: SurgicalMaskRCNNDataset(args.root, split=s) for s in ["train", "val", "test"]}
    train_loader = torch.utils.data.DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collate_fn)
    cat_name = {c["id"]: c["name"] for c in ds["train"].coco.loadCats(ds["train"].coco.getCatIds())}
    idx_to_name = {idx: cat_name[cid] for cid, idx in ds["train"].cat_id_to_idx.items()}
    gt = {s: load_coco_gt(s, args.root) for s in ["train", "val", "test"]}

    rng = random.Random(args.seed)
    train_eval_idx = list(range(len(ds["train"])))
    if args.train_eval_size:
        train_eval_idx = sorted(rng.sample(train_eval_idx, args.train_eval_size))

    model = get_model(len(ds["train"].cat_ids) + 1, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

    curve_path = RESULT / "data" / "training_curves" / "maskrcnn_rerun.csv"
    rows, best_ap, best_ckpt = [], -1.0, args.output_dir / "maskrcnn_resnet50_fpn_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        scheduler.step()
        val_loss = loss_on_split(model, val_loader, device)
        val_pred = {"images": predict_split(model, ds["val"], device, args.score_threshold, idx_to_name)}
        tr_pred = {"images": predict_split(model, ds["train"], device, args.score_threshold, idx_to_name, train_eval_idx)}
        v = evaluate_predictions(val_pred, gt["val"])["aggregate"]
        t = evaluate_predictions(tr_pred, gt["train"])["aggregate"]
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_precision": t["precision"], "train_recall": t["recall"],
               "val_precision": v["precision"], "val_recall": v["recall"],
               "val_mAP50": v["mAP50"], "val_mAP50_95": v["mAP50_95"]}
        rows.append(row)
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
        with open(curve_path, "w", newline="") as f:      # rewrite every epoch so partial runs are usable
            w = csv.DictWriter(f, fieldnames=CURVE_COLS); w.writeheader()
            for r in rows:
                w.writerow({k: ("n/p" if r[k] is None else f"{r[k]:.6f}" if k != "epoch" else r[k]) for k in CURVE_COLS})
        if v["mAP50_95"] is not None and v["mAP50_95"] > best_ap:
            best_ap = v["mAP50_95"]; torch.save(model.state_dict(), best_ckpt)
            print(f"  new best val mask mAP50-95 {best_ap:.4f} -> {best_ckpt}")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_images = predict_split(model, ds["test"], device, args.score_threshold, idx_to_name)
    out = RESULT / "data" / "predictions" / ("maskrcnn_test.json" if args.adopt else "maskrcnn_rerun_test.json")
    save_predictions(out, "maskrcnn_rerun", "test", args.score_threshold, test_images)
    (args.output_dir / "recipe.txt").write_text(
        f"epochs={args.epochs} lr={args.lr} lr_steps={args.lr_steps} batch={args.batch_size} seed={args.seed}\n")
    print("wrote", curve_path, "and", out)


if __name__ == "__main__":
    main()
