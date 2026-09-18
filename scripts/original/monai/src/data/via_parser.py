"""Parse VIA (VGG Image Annotator) JSON format for video annotations."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ClassConfig


@dataclass
class PolygonAnnotation:
    """Single polygon annotation."""
    timestamp: float  # seconds
    class_name: str
    class_id: int
    vertices: np.ndarray  # Nx2 array of (x, y) points


@dataclass
class FrameAnnotations:
    """All annotations for a single frame."""
    timestamp: float
    frame_id: str  # formatted frame ID (e.g., "frame_000123")
    polygons: List[PolygonAnnotation]


class VIAParser:
    """Parse VIA JSON annotation files."""

    def __init__(self, class_config: Optional[ClassConfig] = None):
        self.class_config = class_config or ClassConfig()

    def parse_file(self, json_path: Path) -> List[FrameAnnotations]:
        """Parse a VIA JSON file and return frame annotations."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Group annotations by timestamp
        timestamp_annotations: Dict[float, List[PolygonAnnotation]] = {}

        for key, metadata in data.get('metadata', {}).items():
            annotation = self._parse_metadata(metadata)
            if annotation is None:
                continue

            ts = annotation.timestamp
            if ts not in timestamp_annotations:
                timestamp_annotations[ts] = []
            timestamp_annotations[ts].append(annotation)

        # Convert to FrameAnnotations
        frames = []
        for timestamp in sorted(timestamp_annotations.keys()):
            frame_id = self._timestamp_to_frame_id(timestamp)
            frames.append(FrameAnnotations(
                timestamp=timestamp,
                frame_id=frame_id,
                polygons=timestamp_annotations[timestamp]
            ))

        return frames

    def _parse_metadata(self, metadata: Dict) -> Optional[PolygonAnnotation]:
        """Parse a single metadata entry."""
        # Get timestamp from z field
        z = metadata.get('z', [])
        if not z:
            return None
        timestamp = float(z[0])

        # Get class name from av field
        av = metadata.get('av', {})
        if not av:
            return None
        # av is like {'2': 'Tube'} - get the first value
        class_name = list(av.values())[0] if av else None
        if not class_name:
            return None

        # Resolve class ID
        class_id = self.class_config.get_class_id(class_name)

        # Get polygon vertices from xy field
        # Format: [6, x1, y1, x2, y2, ...] where 6 indicates polygon
        xy = metadata.get('xy', [])
        if len(xy) < 7:  # Need at least type + 3 points
            return None

        # First element is shape type (6 = polygon), rest are x,y pairs
        coords = xy[1:]
        if len(coords) % 2 != 0:
            return None

        vertices = np.array(coords).reshape(-1, 2)

        return PolygonAnnotation(
            timestamp=timestamp,
            class_name=class_name,
            class_id=class_id,
            vertices=vertices
        )

    def _timestamp_to_frame_id(self, timestamp: float) -> str:
        """Convert timestamp to frame ID string."""
        # Use milliseconds for uniqueness
        ms = int(timestamp * 1000)
        return f"frame_{ms:06d}"

    def parse_multiple_files(self, json_paths: List[Path]) -> List[FrameAnnotations]:
        """Parse multiple JSON files and merge annotations."""
        all_frames: Dict[float, FrameAnnotations] = {}

        for json_path in json_paths:
            frames = self.parse_file(json_path)
            for frame in frames:
                if frame.timestamp in all_frames:
                    # Merge polygons for same timestamp
                    all_frames[frame.timestamp].polygons.extend(frame.polygons)
                else:
                    all_frames[frame.timestamp] = frame

        return sorted(all_frames.values(), key=lambda f: f.timestamp)

    def get_unique_timestamps(self, json_paths: List[Path]) -> List[float]:
        """Get all unique timestamps from annotation files."""
        timestamps = set()
        for json_path in json_paths:
            frames = self.parse_file(json_path)
            for frame in frames:
                timestamps.add(frame.timestamp)
        return sorted(timestamps)

    def get_class_statistics(self, frames: List[FrameAnnotations], resolve_aliases: bool = True) -> Dict[str, int]:
        """Get annotation count per class.

        Args:
            frames: List of frame annotations
            resolve_aliases: If True, merge aliases (e.g., 'Suction Tube' -> 'Suction')
        """
        counts: Dict[str, int] = {}
        for frame in frames:
            for poly in frame.polygons:
                name = poly.class_name
                if resolve_aliases:
                    name = self.class_config.resolve_class_name(name)
                counts[name] = counts.get(name, 0) + 1
        return counts
