#!/usr/bin/env python3
"""
Rasterize the COCO 1200images annotations into single-channel semantic masks
for the MONAI UNet, so it trains on EXACTLY the same polygons, class
vocabulary (21 classes + background) and split as the other three models.

  python scripts/rerun/monai/prepare_masks.py          # CPU, ~1 min, idempotent

Writes  scripts/rerun/monai/data/
          masks/<split>/<file_name>.png    uint8, 0 = background, k = category k (see classes.json)
          classes.json                     {"id_to_class": {...}, "class_to_id": {...}}   (0..21)
          split.json                       {"train": [...file names...], "val": [...], "test": [...]}
Frames are read directly from processed_data/coco_format/1200images/images/<split>/ (no copy).

Overlapping instances: drawn largest-area first, so a small instance that sits
on top of a big one wins the pixel (e.g. an instrument over the soft palate).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2]
sys.path.insert(0, str(RESULT / "scripts"))
from seglib import COCO_1200  # noqa: E402

OUT = HERE / "data"


def main():
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    split, id_to_class = {}, None
    for s in ["train", "val", "test"]:
        coco = json.load(open(COCO_1200 / "annotations" / f"instances_{s}.json"))
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        if id_to_class is None:
            id_to_class = {0: "background", **{i + 1: c["name"] for i, c in enumerate(cats)}}
        cat_to_idx = {c["id"]: i + 1 for i, c in enumerate(cats)}
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        (OUT / "masks" / s).mkdir(exist_ok=True)
        files = []
        for im in sorted(coco["images"], key=lambda x: x["file_name"]):
            m = np.zeros((im["height"], im["width"]), dtype=np.uint8)
            for a in sorted(anns.get(im["id"], []), key=lambda a: -a.get("area", 0)):
                if a.get("iscrowd", 0) or not isinstance(a.get("segmentation"), list):
                    continue
                for poly in a["segmentation"]:
                    if len(poly) < 6:
                        continue
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                    cv2.fillPoly(m, [pts], int(cat_to_idx[a["category_id"]]))
            cv2.imwrite(str(OUT / "masks" / s / (Path(im["file_name"]).stem + ".png")), m)
            files.append(im["file_name"])
        split[s] = files
        print(f"{s}: {len(files)} masks")
    json.dump({"id_to_class": id_to_class, "class_to_id": {v: k for k, v in id_to_class.items()}},
              open(OUT / "classes.json", "w"), indent=1)
    json.dump(split, open(OUT / "split.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
