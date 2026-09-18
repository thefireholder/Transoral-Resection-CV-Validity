#!/usr/bin/env python3
"""
Train + test Mask R-CNN on one training subset with the same recipe as
scripts/rerun/maskrcnn_train_curves.py (12 epochs, LR x0.1 after 8 and 11,
bs 4, lr 0.005, seed 42) and score the test split with seglib (COCO AP rule,
matched-instance IoU / Dice).
  python maskrcnn_sweep.py --n 150 [--seed 42] [--epochs 12 --lr-steps 8 11]
Appends one row to data/dataset_size_sweep.csv (via collect_sweep.py).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "scripts"))
sys.path.insert(0, str(RESULT / "scripts" / "original" / "maskrcnn"))
sys.path.insert(0, str(RESULT / "scripts" / "rerun"))
from seglib import evaluate_predictions, load_coco_gt, save_predictions  # noqa: E402
from train_maskrcnn import SurgicalMaskRCNNDataset, collate_fn, get_model, set_seed, train_one_epoch  # noqa: E402
from maskrcnn_train_curves import predict_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr-steps", type=int, nargs="*", default=[8, 11])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()

    root = HERE / "subsets" / "coco" / f"n{args.n}"
    run_dir = HERE / "runs" / "maskrcnn" / f"n{args.n}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = {s: SurgicalMaskRCNNDataset(root, split=s) for s in ["train", "val", "test"]}
    loader = torch.utils.data.DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, collate_fn=collate_fn)
    cat_name = {c["id"]: c["name"] for c in ds["train"].coco.loadCats(ds["train"].coco.getCatIds())}
    idx_to_name = {idx: cat_name[cid] for cid, idx in ds["train"].cat_id_to_idx.items()}
    gt = {s: load_coco_gt(s, root) for s in ["val", "test"]}

    model = get_model(len(ds["train"].cat_ids) + 1, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps, gamma=0.1)
    best, best_ckpt = -1.0, run_dir / "best.pth"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, opt, loader, device); sched.step()
        v = evaluate_predictions({"images": predict_split(model, ds["val"], device, 0.05, idx_to_name)}, gt["val"])["aggregate"]
        print(f"epoch {epoch} loss {loss:.4f} val mAP50-95 {v['mAP50_95']}", flush=True)
        if v["mAP50_95"] is not None and v["mAP50_95"] > best:
            best = v["mAP50_95"]; torch.save(model.state_dict(), best_ckpt)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_images = predict_split(model, ds["test"], device, 0.05, idx_to_name)
    save_predictions(run_dir / "predictions_test.json", "maskrcnn", "test", 0.05, test_images)
    a = evaluate_predictions({"images": test_images}, gt["test"])["aggregate"]
    row = {"model": "maskrcnn", "split": "test", "seed": args.seed, "n_train": args.n,
           "mAP50": a["mAP50"], "mAP50_95": a["mAP50_95"], "precision": a["precision"], "recall": a["recall"],
           "iou": a["iou"], "dice": a["dice"], "run_dir": str(run_dir)}
    json.dump(row, open(run_dir / "sweep_row.json", "w"), indent=1)
    print(row)
    sys.path.insert(0, str(HERE)); import collect_sweep; collect_sweep.main()


if __name__ == "__main__":
    main()
