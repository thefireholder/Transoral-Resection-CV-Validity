#!/usr/bin/env python3
"""
Build seeded, NESTED subsets of the training split for the dataset-size
sweep (Figure 5): 75 c 150 c 300 c 600 c 732(all) images.

  python scripts/rerun/dataset_size_sweep/make_subsets.py            # sizes from plot_config.FIG5_SIZES

Writes under scripts/rerun/dataset_size_sweep/subsets/:
  yolo/n<N>/data.yaml        (train = txt list of image paths; val/test = the normal split)
  coco/n<N>/annotations/instances_{train,val,test}.json  + images -> symlink to the full set
  subsets.json               which files are in which subset

NOTE: the full training split has 732 images (1048 total = 732/157/158),
so the requested sizes 900 and 1200 cannot be reached; they are clipped to
732 and written as n732 (the sweep csv records the real n_train).
Val and test splits are never subsampled. The random subset is seeded
(--seed 0) and nested, so a larger size always contains the smaller one.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "plot"))
import plot_config as C  # noqa: E402

sys.path.insert(0, str(RESULT / "scripts"))
from seglib import COCO_1200 as COCO_ROOT, YOLO_1200 as YOLO_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=C.FIG5_SIZES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco_train = json.load(open(COCO_ROOT / "annotations" / "instances_train.json"))
    all_files = sorted(im["file_name"] for im in coco_train["images"])
    n_all = len(all_files)
    order = all_files[:]
    random.Random(args.seed).shuffle(order)

    manifest = {"seed": args.seed, "n_train_total": n_all, "subsets": {}}
    for n_req in sorted(set(args.sizes)):
        n = min(n_req, n_all)
        files = sorted(order[:n])
        keep = set(files)
        manifest["subsets"][f"n{n}"] = {"requested": n_req, "n_train": n, "files": files}

        # ---- YOLO ----
        ydir = HERE / "subsets" / "yolo" / f"n{n}"
        ydir.mkdir(parents=True, exist_ok=True)
        (ydir / "train.txt").write_text("\n".join(str(YOLO_ROOT / "images" / "train" / f) for f in files) + "\n")
        import yaml
        data_yaml = yaml.safe_load(open(YOLO_ROOT / "data.yaml"))
        data_yaml["train"] = str(ydir / "train.txt")
        data_yaml["val"] = str(YOLO_ROOT / "images" / "val")
        data_yaml["test"] = str(YOLO_ROOT / "images" / "test")
        data_yaml["path"] = str(YOLO_ROOT)
        yaml.safe_dump(data_yaml, open(ydir / "data.yaml", "w"), sort_keys=False)

        # ---- COCO (Mask R-CNN) ----
        cdir = HERE / "subsets" / "coco" / f"n{n}"
        (cdir / "annotations").mkdir(parents=True, exist_ok=True)
        sub = dict(coco_train)
        sub["images"] = [im for im in coco_train["images"] if im["file_name"] in keep]
        ids = {im["id"] for im in sub["images"]}
        sub["annotations"] = [a for a in coco_train["annotations"] if a["image_id"] in ids]
        json.dump(sub, open(cdir / "annotations" / "instances_train.json", "w"))
        for s in ["val", "test"]:
            shutil.copy(COCO_ROOT / "annotations" / f"instances_{s}.json", cdir / "annotations")
        if not (cdir / "images").exists():
            (cdir / "images").symlink_to(COCO_ROOT / "images")
        print(f"n{n}: {n} train images ({len(sub['annotations'])} annotations)")

    json.dump(manifest, open(HERE / "subsets" / "subsets.json", "w"), indent=1)
    print("wrote", HERE / "subsets" / "subsets.json")


if __name__ == "__main__":
    main()
