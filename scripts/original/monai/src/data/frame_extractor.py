"""Extract frames from video at specified timestamps."""
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm


class FrameExtractor:
    """Extract frames from video file."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None
        self._frame_count: Optional[int] = None
        self._frame_size: Optional[Tuple[int, int]] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open video file."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def close(self):
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._fps is None:
            raise RuntimeError("Video not opened")
        return self._fps

    @property
    def frame_count(self) -> int:
        if self._frame_count is None:
            raise RuntimeError("Video not opened")
        return self._frame_count

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height)."""
        if self._frame_size is None:
            raise RuntimeError("Video not opened")
        return self._frame_size

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self.frame_count / self.fps

    def timestamp_to_frame_number(self, timestamp: float) -> int:
        """Convert timestamp (seconds) to frame number."""
        return int(timestamp * self.fps)

    def extract_frame_at_timestamp(self, timestamp: float) -> Optional[cv2.Mat]:
        """Extract a single frame at the given timestamp."""
        if self._cap is None:
            raise RuntimeError("Video not opened")

        frame_number = self.timestamp_to_frame_number(timestamp)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self._cap.read()
        if not ret:
            return None

        return frame

    def extract_frames_at_timestamps(
        self,
        timestamps: List[float],
        output_dir: Path,
        show_progress: bool = True
    ) -> List[Path]:
        """Extract frames at specified timestamps and save to output directory.

        Args:
            timestamps: List of timestamps in seconds
            output_dir: Directory to save extracted frames
            show_progress: Show progress bar

        Returns:
            List of paths to saved frames
        """
        if self._cap is None:
            raise RuntimeError("Video not opened")

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        # Sort timestamps for sequential access (more efficient)
        sorted_timestamps = sorted(set(timestamps))

        iterator = tqdm(sorted_timestamps, desc="Extracting frames") if show_progress else sorted_timestamps

        for timestamp in iterator:
            frame = self.extract_frame_at_timestamp(timestamp)
            if frame is None:
                print(f"Warning: Could not extract frame at {timestamp}s")
                continue

            # Generate frame ID from timestamp (milliseconds)
            frame_id = int(timestamp * 1000)
            output_path = output_dir / f"frame_{frame_id:06d}.png"

            # Convert BGR to RGB before saving (OpenCV uses BGR)
            # Actually, for PNG we can keep BGR as cv2.imwrite expects BGR
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(output_path)

        return saved_paths


def extract_annotated_frames(
    video_path: Path,
    timestamps: List[float],
    output_dir: Path
) -> List[Path]:
    """Convenience function to extract frames at timestamps."""
    with FrameExtractor(video_path) as extractor:
        print(f"Video: {video_path.name}")
        print(f"  Resolution: {extractor.frame_size[0]}x{extractor.frame_size[1]}")
        print(f"  FPS: {extractor.fps:.2f}")
        print(f"  Duration: {extractor.duration:.2f}s")
        print(f"  Extracting {len(timestamps)} frames...")

        return extractor.extract_frames_at_timestamps(timestamps, output_dir)
