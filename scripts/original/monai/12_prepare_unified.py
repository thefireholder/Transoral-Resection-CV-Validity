#!/usr/bin/env python3
"""Script 12: Build the unified two-case dataset (JSR + JML619).

Schema and decisions: data/unified_schema.md. Output:
  data/unified/frames/{jsr_,jml_}frame_NNNNN.jpg   (hardlinks to source frames)
  data/unified/masks/{jsr_,jml_}frame_NNNNN.png    (unified class IDs)
  data/unified/split.json                          (union of team temporal splits)

Requires scripts 05 (JSR frames) and 08 (JML frames) to have been run.
"""
import sys
import json
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, UnifiedClassConfig, AnatomyClassConfig
from src.data.anatomy_parser import AnatomyParser
from src.data.via_video_parser import ViaVideoParser, snap_to_split
from src.data.mask_generator import MaskGenerator
from src.training.losses import (
    compute_class_weights_from_masks,
    print_class_weight_report,
)

FRAME_SIZE = (1920, 1080)  # both source videos

# JSR-canonical class name (AnatomyClassConfig) -> unified class name.
# See data/unified_schema.md D1-D5.
JSR_TO_UNIFIED = {
    "BOT": "base of tongue",
    "Soft Palate": "soft palate",
    "Pharyngeal": "posterior pharyngeal wall",
    "Pterygoid": "medial pterygoid muscle",
    "Fat": "fat fascia",
    "Cut": "cut",
    "Bipolar": "maryland",
    "Monopolar": "monopolar",
    "Suction": "suction",
    "Endotracheal Tube": "endotracheal tube",
    "Ligasure": "ligasure",
}


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    config = get_config()
    ucfg = UnifiedClassConfig()
    frames_out = config.paths.unified_frames_dir
    masks_out = config.paths.unified_masks_dir
    frames_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    gen = MaskGenerator(FRAME_SIZE, ucfg.num_classes)
    from typing import cast, Any

    print("=" * 60)
    print("Step 12: Build Unified Dataset (JSR + JML619)")
    print("=" * 60)

    # ---------------- JSR ------------------------------------------------
    print("\n[JSR] parsing annotations...")
    acfg = AnatomyClassConfig()
    jsr_frames = AnatomyParser(acfg).parse_directory(config.paths.anatomy_dir)
    dropped = 0
    for f in jsr_frames:
        kept = []
        for p in f.polygons:
            # parser stores the raw option string; resolve to JSR-canonical first
            uname = JSR_TO_UNIFIED.get(acfg.resolve_class_name(p.class_name))
            if uname is None:
                dropped += 1
                continue
            p.class_name = uname
            p.class_id = ucfg.class_to_id[uname]
            kept.append(p)
        f.polygons = kept
        f.frame_id = f"jsr_frame_{f.frame_idx:05d}"
    jsr_frames = [f for f in jsr_frames if f.polygons]
    print(f"[JSR] {len(jsr_frames)} frames, "
          f"{sum(len(f.polygons) for f in jsr_frames)} polygons"
          + (f", {dropped} polygons with unmapped labels dropped" if dropped else ""))

    jsr_src = config.paths.anatomy_frames_dir
    missing = [f.frame_idx for f in jsr_frames
               if not (jsr_src / f"frame_{f.frame_idx:05d}.jpg").exists()]
    if missing:
        raise SystemExit(f"[JSR] {len(missing)} frames missing from {jsr_src} "
                         f"(run scripts/05 first): {missing[:5]}...")
    for f in jsr_frames:
        link(jsr_src / f"frame_{f.frame_idx:05d}.jpg",
             frames_out / f"{f.frame_id}.jpg")

    # ---------------- JML ------------------------------------------------
    print("\n[JML] parsing annotations...")
    import cv2
    cap = cv2.VideoCapture(str(config.paths.jml_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    parser = ViaVideoParser(ucfg, merge_map_path=config.paths.class_merge_file)
    jml_frames, stats = parser.parse(
        config.paths.jml_annotation_file,
        video_fname=config.paths.jml_video_path.name, fps=fps)
    with open(config.paths.jml_split_file) as f:
        jml_team_split = json.load(f)
    jml_split_stems = {name: [Path(p).stem for p in paths]
                       for name, paths in jml_team_split.items()}
    split_indices = {int(s[len("frame_"):])
                     for stems in jml_split_stems.values() for s in stems}
    snap_to_split(jml_frames, split_indices)
    if stats["unknown_labels"]:
        print(f"[JML] WARNING unmapped labels: {dict(stats['unknown_labels'])}")
    print(f"[JML] {len(jml_frames)} frames, "
          f"{sum(stats['classes'].values())} polygons")

    jml_src = config.paths.jml_frames_dir
    for f in jml_frames:
        src = jml_src / f"{f.frame_id}.jpg"
        if not src.exists():
            raise SystemExit(f"[JML] missing frame {src} (run scripts/08 first)")
        f.frame_id = f"jml_{f.frame_id}"
        link(src, frames_out / f"{f.frame_id}.jpg")

    # ---------------- masks ----------------------------------------------
    all_frames = jsr_frames + jml_frames
    print(f"\nGenerating {len(all_frames)} masks -> {masks_out}")
    gen.generate_masks(cast(Any, all_frames), masks_out)

    # ---------------- split ----------------------------------------------
    with open(config.paths.jsr_split_file) as f:
        jsr_team_split = json.load(f)
    combined = {}
    for name in ("train", "val", "test"):
        entries = []
        for p in jsr_team_split[name]:
            fn = f"jsr_{Path(p).stem}.jpg"
            if (frames_out / fn).exists():
                entries.append(fn)
        for stem in jml_split_stems[name]:
            fn = f"jml_{stem}.jpg"
            if (frames_out / fn).exists():
                entries.append(fn)
        combined[name] = sorted(entries)
    with open(config.paths.unified_split_file, "w") as f:
        json.dump(combined, f, indent=2)
    counts = {k: len(v) for k, v in combined.items()}
    per_case = {k: (sum(e.startswith('jsr_') for e in v),
                    sum(e.startswith('jml_') for e in v))
                for k, v in combined.items()}
    print(f"\nSplit {counts}  (jsr, jml) per split: {per_case}")

    # ---------------- class report ----------------------------------------
    print("\nClass distribution / weights over unified masks:")
    weights, wstats = compute_class_weights_from_masks(masks_out, ucfg)
    print_class_weight_report(weights, wstats, ucfg)

    print("\n" + "=" * 60)
    print("Unified dataset ready — see data/unified_schema.md for the schema.")
    print("Next: python scripts/13_train_unified.py")


if __name__ == "__main__":
    main()
