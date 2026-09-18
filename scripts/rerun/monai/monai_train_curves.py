#!/usr/bin/env python3
"""
Train the MONAI UNet of train_script/monai-project on the COCO-derived masks
(scripts/rerun/monai/prepare_masks.py) -- same 21-class vocabulary, same
732/157/158 split as the other three models -- with the project's own recipe:

  UNet(channels 32-64-128-256-512, strides 2x4, 2 res units, dropout 0.2), input 512x512,
  DiceFocal loss (0.5/0.5, gamma 2) with inverse-frequency class weights from the masks,
  AdamW lr 1e-4 wd 1e-5, CosineAnnealingLR(T_max=100, eta_min=1e-6), grad-clip 1.0,
  batch 4, up to 100 epochs, early stopping patience 15 on val mean Dice, seed 42.
  Best checkpoint = highest val semantic mean Dice (the project's convention).

Every epoch it logs, in the uniform training-curve schema, instance-level
mask precision / recall / AP on the val split and on a fixed seeded 157-frame
train subset -- computed by scripts/seglib.py exactly like the other models.
Instances of the semantic output are 8-connected components of the argmax
map, scored by the mean softmax probability of the class over the component
(the convention of the project's scripts/17_compute_instance_ap.py);
components below --min-component-px (on the 512x512 grid) are dropped.

Outputs
  data/training_curves/monai_rerun.csv          (Figure 1a; collect_results.py prefers it)
  data/predictions/monai_test.json              (--adopt, default on) -> Figures 1b, 2, 3, 4 via compute_mask_metrics.py
  scripts/rerun/output/monai/{best_model.pth,final_model.pth,recipe.txt,log.txt}

GPU. ~1-2 min / epoch on a 6 GB laptop GPU (fits at batch 4, 512x512).
  python scripts/rerun/monai/monai_train_curves.py            # full recipe
  python scripts/rerun/monai/monai_train_curves.py --eval-only scripts/rerun/output/monai/best_model.pth
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "scripts"))
sys.path.insert(0, str(RESULT / "scripts" / "original" / "monai"))
from seglib import COCO_1200, evaluate_predictions, load_coco_gt, rle_encode, save_predictions  # noqa: E402
from src.config import ModelConfig  # noqa: E402
from src.data.dataset import InstrumentDataset  # noqa: E402
from src.models.segmentation import SegmentationModel  # noqa: E402
from src.training.losses import DiceFocalLoss, compute_class_weights_from_masks, print_class_weight_report  # noqa: E402
from src.transforms.augmentations import get_train_transforms, get_val_transforms  # noqa: E402
from monai.metrics import DiceMetric  # noqa: E402

DATA = HERE / "data"
SIZE = 512
CURVE_COLS = ["epoch", "train_loss", "val_loss", "train_precision", "train_recall",
              "val_precision", "val_recall", "val_mAP50", "val_mAP50_95", "val_dice_semantic", "lr"]


class COCOClassConfig:
    """Minimal class config for compute_class_weights_from_masks (0 = background)."""
    def __init__(self, id_to_class):
        self.id_to_class = {int(k): v for k, v in id_to_class.items()}
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.num_classes = len(self.id_to_class)
        self.weight_multipliers = {}
        self.background_weight = 0.1


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_dataset(split, files, transform):
    imgs = [COCO_1200 / "images" / split / f for f in files]
    masks = [DATA / "masks" / split / (Path(f).stem + ".png") for f in files]
    return InstrumentDataset(imgs, masks, transform=transform, target_size=(SIZE, SIZE))


@torch.no_grad()
def predict_instances(model, split, files, device, id_to_class, min_px, batch=8):
    """Uniform prediction entries (full-res RLE) for the given frames."""
    model.eval()
    out = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        xs, shapes = [], []
        for f in chunk:
            img = cv2.imread(str(COCO_1200 / "images" / split / f))
            shapes.append(img.shape[:2])
            x = cv2.cvtColor(cv2.resize(img, (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            xs.append(torch.from_numpy(x.transpose(2, 0, 1)))
        probs = torch.softmax(model(torch.stack(xs).to(device)), dim=1).cpu().numpy()
        for f, (h, w), pr in zip(chunk, shapes, probs):
            pred = pr.argmax(0).astype(np.uint8)
            preds = []
            for c in np.unique(pred):
                if c == 0:
                    continue
                n, lab = cv2.connectedComponents((pred == c).astype(np.uint8), connectivity=8)
                for k in range(1, n):
                    comp = lab == k
                    if comp.sum() < min_px:
                        continue
                    full = cv2.resize(comp.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if not full.any():
                        continue
                    preds.append({"class": id_to_class[int(c)], "score": float(pr[c][comp].mean()),
                                  "rle": rle_encode(full)})
            out.append({"file_name": f, "height": int(h), "width": int(w), "predictions": preds})
    return out


@torch.no_grad()
def val_loss_and_dice(model, loader, criterion, device, n_cls):
    model.eval()
    dm = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    tot, n = 0.0, 0
    for b in loader:
        x, y = b["image"].to(device), b["label"].to(device)
        logits = model(x)
        tot += float(criterion(logits, y)); n += 1
        pred_oh = torch.nn.functional.one_hot(logits.argmax(1), n_cls).permute(0, 3, 1, 2)
        gt_oh = torch.nn.functional.one_hot(y[:, 0], n_cls).permute(0, 3, 1, 2)
        dm(pred_oh, gt_oh)
    return tot / max(n, 1), float(dm.aggregate().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-component-px", type=int, default=20)
    ap.add_argument("--train-eval-size", type=int, default=157, help="0 = evaluate on all 732 train frames")
    ap.add_argument("--eval-only", type=Path, default=None, help="skip training; load this checkpoint")
    ap.add_argument("--no-adopt", action="store_true", help="write monai_rerun_test.json instead of monai_test.json")
    ap.add_argument("--output-dir", type=Path, default=RESULT / "scripts" / "rerun" / "output" / "monai")
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = open(args.output_dir / "log.txt", "a")

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

    say(f"device {device}  torch {torch.__version__}")
    classes = json.load(open(DATA / "classes.json"))
    id_to_class = {int(k): v for k, v in classes["id_to_class"].items()}
    ccfg = COCOClassConfig(id_to_class)
    split = json.load(open(DATA / "split.json"))
    gt = {s: load_coco_gt(s) for s in ["train", "val", "test"]}
    train_eval = split["train"] if not args.train_eval_size else \
        sorted(random.Random(args.seed).sample(split["train"], args.train_eval_size))

    model = SegmentationModel(ModelConfig(out_channels=ccfg.num_classes)).to(device)
    say(f"UNet params {model.get_num_parameters():,}  classes {ccfg.num_classes}")
    curve_path = RESULT / "data" / "training_curves" / "monai_rerun.csv"
    best_ckpt = args.output_dir / "best_model.pth"

    if args.eval_only is None:
        train_loader = torch.utils.data.DataLoader(
            make_dataset("train", split["train"], get_train_transforms()), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            make_dataset("val", split["val"], get_val_transforms()), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers)
        class_weights, wstats = compute_class_weights_from_masks(DATA / "masks" / "train", ccfg)
        print_class_weight_report(class_weights, wstats, ccfg)
        criterion = DiceFocalLoss(num_classes=ccfg.num_classes, class_weights=class_weights.to(device),
                                  dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        rows, best_dice, bad = [], -1.0, 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); tot, n = 0.0, 0
            for b in train_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += float(loss); n += 1
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            train_loss = tot / max(n, 1)
            vloss, vdice = val_loss_and_dice(model, val_loader, criterion, device, ccfg.num_classes)
            v = evaluate_predictions({"images": predict_instances(model, "val", split["val"], device, id_to_class,
                                                                   args.min_component_px)}, gt["val"])["aggregate"]
            t = evaluate_predictions({"images": predict_instances(model, "train", train_eval, device, id_to_class,
                                                                   args.min_component_px)}, gt["train"])["aggregate"]
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": vloss,
                   "train_precision": t["precision"], "train_recall": t["recall"],
                   "val_precision": v["precision"], "val_recall": v["recall"],
                   "val_mAP50": v["mAP50"], "val_mAP50_95": v["mAP50_95"], "val_dice_semantic": vdice, "lr": lr}
            rows.append(row)
            say({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, f"({time.time() - t0:.0f}s)")
            with open(curve_path, "w", newline="") as f:      # rewritten every epoch -> partial runs are usable
                w = csv.DictWriter(f, fieldnames=CURVE_COLS); w.writeheader()
                for r in rows:
                    w.writerow({k: ("n/p" if r[k] is None else (f"{r[k]:.6f}" if k != "epoch" else r[k])) for k in CURVE_COLS})
            if vdice > best_dice:
                best_dice, bad = vdice, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_dice": vdice,
                            "val_mAP50_95": v["mAP50_95"]}, best_ckpt)
                say(f"  new best val mean Dice {vdice:.4f} (epoch {epoch}) -> {best_ckpt}")
            else:
                bad += 1
                if bad >= args.patience:
                    say(f"early stopping at epoch {epoch} (no val Dice improvement for {args.patience} epochs)"); break
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, args.output_dir / "final_model.pth")
        (args.output_dir / "recipe.txt").write_text(
            f"epochs_max={args.epochs} epochs_run={epoch} patience={args.patience} batch={args.batch_size} lr={args.lr} "
            f"wd=1e-5 cosine_T={args.epochs} seed={args.seed} input={SIZE}x{SIZE} min_component_px={args.min_component_px} "
            f"best_epoch={torch.load(best_ckpt, map_location='cpu', weights_only=False)['epoch']} best_val_dice={best_dice:.4f}\n")
        ckpt_to_load = best_ckpt
    else:
        ckpt_to_load = args.eval_only

    ck = torch.load(ckpt_to_load, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    say(f"loaded {ckpt_to_load} (epoch {ck.get('epoch')}, val dice {ck.get('best_dice')})")
    test_images = predict_instances(model, "test", split["test"], device, id_to_class, args.min_component_px)
    out = RESULT / "data" / "predictions" / ("monai_rerun_test.json" if args.no_adopt else "monai_test.json")
    save_predictions(out, "monai_rerun", "test", 0.0, test_images)
    a = evaluate_predictions({"images": test_images}, gt["test"])["aggregate"]
    say(f"test: {{k: round(v, 4) for k, v in a.items() if v is not None}}".replace("{k: round(v, 4) for k, v in a.items() if v is not None}",
        str({k: round(v, 4) for k, v in a.items() if v is not None})))
    say("wrote", out)


if __name__ == "__main__":
    main()
