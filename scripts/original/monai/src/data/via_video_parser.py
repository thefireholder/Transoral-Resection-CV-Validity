"""Parse VIA *video* project annotations (the JML619 delivery format).

This differs from the per-frame image projects handled by anatomy_parser.py:

1. The whole project is one JSON; `file` contains the video itself (plus
   leftover demo entries from the VIA template the annotators started from).
   Each `metadata` entry carries `z: [timestamp_seconds]` instead of pointing
   at a frame image, so frame index = round(z * fps).

2. Class labels are free text under attribute "1" (e.g. "Yankauer Suction",
   "Monopolar cauteryy"), not option-index lookups. Labels are normalized:
   lowercase -> config aliases (typos) -> team merge map
   (class_merge_default.json) -> class_to_id.

Output reuses AnatomyFrameAnnotations/AnatomyPolygon so MaskGenerator works
unchanged.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import JML619ClassConfig
from src.data.anatomy_parser import (
    AnatomyFrameAnnotations,
    AnatomyPolygon,
    FILLABLE_SHAPE_TYPES,
)

# Free-text label attribute id in the VIA project.
LABEL_ATTRIBUTE_ID = "1"


def snap_to_split(frames, split_indices):
    """Align parsed frame indices with a split file's frame numbers (+-1).

    The team's split files were generated with slightly different timestamp
    rounding for a few frames; snapping keeps image/mask/split pairing exact.
    Mutates frames in place; returns (n_snapped, unmatched_indices).
    """
    claimed = {f.frame_idx for f in frames}
    snapped, unmatched = 0, []
    for f in frames:
        if f.frame_idx in split_indices:
            continue
        for candidate in (f.frame_idx - 1, f.frame_idx + 1):
            if candidate in split_indices and candidate not in claimed:
                claimed.discard(f.frame_idx)
                claimed.add(candidate)
                f.frame_idx = candidate
                f.frame_id = f"frame_{candidate:05d}"
                for p in f.polygons:
                    p.frame_idx = candidate
                snapped += 1
                break
        else:
            unmatched.append(f.frame_idx)
    return snapped, unmatched


class ViaVideoParser:
    """Parse a VIA3 video project into per-frame polygon annotations."""

    def __init__(
        self,
        class_config: Optional[JML619ClassConfig] = None,
        merge_map_path: Optional[Path] = None,
    ):
        self.class_config = class_config or JML619ClassConfig()
        self.merge_map: Dict[str, str] = {}
        if merge_map_path is not None:
            with open(merge_map_path) as f:
                self.merge_map = {
                    k.strip().lower(): v.strip().lower()
                    for k, v in json.load(f).items()
                }

    def normalize_label(self, raw: str) -> str:
        label = raw.strip().lower()
        label = self.class_config.resolve_class_name(label)
        return self.merge_map.get(label, label)

    def parse(
        self,
        json_path: Path,
        video_fname: str,
        fps: float,
    ) -> Tuple[List[AnatomyFrameAnnotations], Dict[str, Counter]]:
        """Parse annotations for `video_fname` out of the project.

        Returns (frames sorted by index, stats) where stats has Counters:
        'classes' (kept polygons per class), 'skipped_shapes' (non-fillable
        shape codes), 'unknown_labels' (normalized labels with no class id).
        """
        with open(json_path) as f:
            d = json.load(f)

        # The project keeps VIA's demo entries around; find the real video.
        target_vid = None
        for fid, finfo in d.get("file", {}).items():
            if Path(finfo.get("fname", "")).name == video_fname:
                target_vid = str(fid)
                break
        if target_vid is None:
            raise ValueError(f"{video_fname!r} not found in {json_path}")

        stats = {
            "classes": Counter(),
            "skipped_shapes": Counter(),
            "unknown_labels": Counter(),
        }
        per_frame: Dict[int, AnatomyFrameAnnotations] = {}

        for m in d.get("metadata", {}).values():
            if str(m.get("vid")) != target_vid:
                continue
            z, xy = m.get("z") or [], m.get("xy") or []
            if not z or len(xy) < 2:
                continue

            shape = xy[0]
            # A handful of corrupt entries have a coordinate where the shape
            # code belongs (e.g. 161.64) — treat anything non-integral or
            # non-fillable as a skip, not an error.
            if shape != int(shape) or int(shape) not in FILLABLE_SHAPE_TYPES:
                stats["skipped_shapes"][shape] += 1
                continue

            coords = xy[1:]
            if len(coords) < 6 or len(coords) % 2 != 0:
                stats["skipped_shapes"]["degenerate"] += 1
                continue

            raw_label = m.get("av", {}).get(LABEL_ATTRIBUTE_ID, "")
            label = self.normalize_label(raw_label)
            if label == "background":
                continue
            if label not in self.class_config.class_to_id:
                stats["unknown_labels"][raw_label] += 1
                continue

            frame_idx = round(z[0] * fps)
            frame = per_frame.setdefault(
                frame_idx,
                AnatomyFrameAnnotations(
                    frame_idx=frame_idx, frame_id=f"frame_{frame_idx:05d}"
                ),
            )
            frame.polygons.append(
                AnatomyPolygon(
                    frame_idx=frame_idx,
                    class_name=label,
                    class_id=self.class_config.class_to_id[label],
                    vertices=np.array(coords, dtype=np.float64).reshape(-1, 2),
                )
            )
            stats["classes"][label] += 1

        return [per_frame[k] for k in sorted(per_frame)], stats
