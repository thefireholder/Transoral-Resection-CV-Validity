"""Parse the anatomy-annotation VIA project files.

The anatomy JSONs differ from the instrument ones in two ways:

1. Each JSON is a multi-frame VIA project. The `file` dict maps file IDs (vid)
   to image filenames like `frame_00100.jpg`. Each `metadata` entry references a
   `vid` to indicate which frame's polygon it belongs to.

2. Class labels are stored as option-index lookups. `metadata.av` looks like
   `{"3": "0"}` meaning "attribute 3, option 0". The actual class name lives in
   `attribute["3"].options["0"]`.

Polylines (shape type 7) and other non-polygon shapes are skipped — segmentation
masks need filled regions.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import AnatomyClassConfig

# VIA shape type codes. We accept polygons (6) and polylines (7) — the
# 10000-10200 batch was annotated with polylines, but cv2.fillPoly closes them
# implicitly, so they rasterize the same way as closed polygons.
FILLABLE_SHAPE_TYPES = {6, 7}


@dataclass
class AnatomyPolygon:
    """A single anatomy polygon annotation, tagged with its frame index."""
    frame_idx: int
    class_name: str
    class_id: int
    vertices: np.ndarray  # Nx2 (x, y)


@dataclass
class AnatomyFrameAnnotations:
    """All polygon annotations for a single frame."""
    frame_idx: int
    frame_id: str  # filename stem, e.g. "frame_00100"
    polygons: List[AnatomyPolygon] = field(default_factory=list)


class AnatomyParser:
    """Parse anatomy VIA project files (multi-frame, option-index encoded)."""

    def __init__(self, class_config: Optional[AnatomyClassConfig] = None):
        self.class_config = class_config or AnatomyClassConfig()

    def parse_directory(self, root: Path) -> List[AnatomyFrameAnnotations]:
        """Walk every JSON under `root`, merging polygons by frame index."""
        per_frame: Dict[int, AnatomyFrameAnnotations] = {}
        for jp in sorted(Path(root).rglob("*.json")):
            try:
                self._parse_into(jp, per_frame)
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse {jp}: {e}")
        return [per_frame[k] for k in sorted(per_frame.keys())]

    def _parse_into(
        self,
        json_path: Path,
        out: Dict[int, AnatomyFrameAnnotations],
    ) -> None:
        with open(json_path) as f:
            d = json.load(f)

        # Build vid -> frame index map
        vid_to_idx: Dict[str, int] = {}
        for fid, finfo in d.get("file", {}).items():
            fname = finfo.get("fname", "")
            if fname.startswith("frame_") and fname.endswith(".jpg"):
                try:
                    vid_to_idx[fid] = int(fname[len("frame_") : -len(".jpg")])
                except ValueError:
                    continue

        # Build attribute_id -> {option_id -> class_name} map
        attr_options: Dict[str, Dict[str, str]] = {}
        for aid, ainfo in d.get("attribute", {}).items():
            attr_options[aid] = ainfo.get("options", {}) or {}

        # Walk metadata
        for m in d.get("metadata", {}).values():
            vid = m.get("vid")
            if vid not in vid_to_idx:
                continue
            xy = m.get("xy", [])
            if not xy or len(xy) < 7:
                continue
            # First element is shape type; only fillable shapes are usable as masks
            if not isinstance(xy[0], (int, float)) or int(xy[0]) not in FILLABLE_SHAPE_TYPES:
                continue

            coords = xy[1:]
            if len(coords) % 2 != 0:
                continue
            vertices = np.array(coords, dtype=np.float64).reshape(-1, 2)
            if len(vertices) < 3:
                continue  # need at least a triangle

            # Resolve class name from av
            av = m.get("av", {})
            class_name: Optional[str] = None
            for aid, opt_id in av.items():
                opts = attr_options.get(aid, {})
                cand = opts.get(opt_id)
                if cand:
                    class_name = cand
                    break
            if not class_name:
                continue

            class_id = self.class_config.get_class_id(class_name)
            if class_id == 0:  # unknown class -> falls back to Background; skip
                continue

            frame_idx = vid_to_idx[vid]
            entry = out.get(frame_idx)
            if entry is None:
                entry = AnatomyFrameAnnotations(
                    frame_idx=frame_idx,
                    frame_id=f"frame_{frame_idx:05d}",
                )
                out[frame_idx] = entry
            entry.polygons.append(AnatomyPolygon(
                frame_idx=frame_idx,
                class_name=class_name,
                class_id=class_id,
                vertices=vertices,
            ))

    def get_class_statistics(
        self,
        frames: List[AnatomyFrameAnnotations],
    ) -> Dict[str, int]:
        """Polygon counts per resolved class name."""
        counts: Dict[str, int] = {}
        for fa in frames:
            for p in fa.polygons:
                resolved = self.class_config.resolve_class_name(p.class_name)
                counts[resolved] = counts.get(resolved, 0) + 1
        return counts
