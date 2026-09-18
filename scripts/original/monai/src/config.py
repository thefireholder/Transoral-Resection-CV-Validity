"""Configuration for surgical instrument segmentation pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PathConfig:
    """Paths configuration."""
    root: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def video_path(self) -> Path:
        return self.data_dir / "JS R Tonsil GSW.mp4"

    @property
    def annotation_files(self) -> List[Path]:
        return [
            self.data_dir / "Instruments_tag_v3.json",
            self.data_dir / "Instrument_tag_vignesh_v4.json",
            self.data_dir / "needle-driver-and-retractor.json",
        ]

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    # Anatomy pipeline (separate from instruments — different annotators,
    # different class set, different JSON format).
    @property
    def anatomy_dir(self) -> Path:
        return self.data_dir / "anatomy"

    @property
    def anatomy_frames_dir(self) -> Path:
        return self.anatomy_dir / "frames"

    @property
    def anatomy_masks_dir(self) -> Path:
        return self.anatomy_dir / "masks"

    @property
    def anatomy_checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints_anatomy"

    @property
    def anatomy_logs_dir(self) -> Path:
        return self.outputs_dir / "logs_anatomy"

    # JML619 pipeline — second annotated case ("JM L Tonsil GSW"), delivered as
    # a single VIA *video* project rather than per-frame image projects.
    @property
    def jml_video_path(self) -> Path:
        return self.root / "TORS-selected" / "JM L Tonsil GSW.mp4"

    @property
    def jml_annotation_file(self) -> Path:
        return self.root / "JML 619 completed frames.json"

    @property
    def jml_split_file(self) -> Path:
        return self.root / "JML619-split.json"

    @property
    def class_merge_file(self) -> Path:
        return self.root / "class_merge_default.json"

    @property
    def jml_dir(self) -> Path:
        return self.data_dir / "jml619"

    @property
    def jml_frames_dir(self) -> Path:
        return self.jml_dir / "frames"

    @property
    def jml_masks_dir(self) -> Path:
        return self.jml_dir / "masks"

    @property
    def jml_resolved_split_file(self) -> Path:
        return self.jml_dir / "split.json"

    @property
    def jml_checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints_jml619"

    @property
    def jml_logs_dir(self) -> Path:
        return self.outputs_dir / "logs_jml619"

    # Unified two-case dataset (JSR + JML619) — see data/unified_schema.md.
    @property
    def unified_dir(self) -> Path:
        return self.data_dir / "unified"

    @property
    def unified_frames_dir(self) -> Path:
        return self.unified_dir / "frames"

    @property
    def unified_masks_dir(self) -> Path:
        return self.unified_dir / "masks"

    @property
    def unified_split_file(self) -> Path:
        return self.unified_dir / "split.json"

    @property
    def jsr_split_file(self) -> Path:
        return self.root / "JSR-split.json"

    @property
    def unified_checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints_unified"

    @property
    def unified_logs_dir(self) -> Path:
        return self.outputs_dir / "logs_unified"


@dataclass
class ClassConfig:
    """Class labels and mappings."""
    # Class name to ID mapping
    class_to_id: Dict[str, int] = field(default_factory=lambda: {
        "Background": 0,
        "Maryland": 1,
        "Cauterizer": 2,
        "Tube": 3,
        "Suction": 4,
        "Retractor": 5,
        "Needle Driver": 6,
    })

    # Aliases for class names (handle variations in annotations)
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "Suction Tube": "Suction",
        "suction": "Suction",
        "maryland": "Maryland",
        "cauterizer": "Cauterizer",
        "tube": "Tube",
        "Endotracheal tube": "Tube",
        "endotracheal tube": "Tube",
        "retractor": "Retractor",
        "needle driver": "Needle Driver",
        "Needle driver": "Needle Driver",
        "default": "Background",
    })

    # Fallback class weights, used only when masks aren't available for live
    # computation (e.g. unit tests). The training pipeline normally derives
    # weights from the mask distribution at startup — see
    # `compute_class_weights_from_masks` in src/training/losses.py.
    class_weights: List[float] = field(default_factory=lambda: [
        0.1, 0.16, 0.23, 1.09, 0.91, 3.0, 7.30,
    ])

    # Per-class multipliers applied on top of the data-derived inverse-frequency
    # weights. Use this for classes that are visually hard and collapse to
    # background even with their natural inverse-frequency weight (e.g.
    # Retractor here is white fabric and was found to need ~2.6x the
    # data-derived weight to learn at all).
    weight_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "Retractor": 2.6,
    })

    # Background gets clamped to a small fixed weight regardless of the
    # data-derived value, since it dominates the pixel count and would otherwise
    # produce vanishing weights for everything else.
    background_weight: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def id_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}

    def resolve_class_name(self, name: str) -> str:
        """Resolve class name aliases."""
        return self.aliases.get(name, name)

    def get_class_id(self, name: str) -> int:
        """Get class ID from name, handling aliases."""
        resolved = self.resolve_class_name(name)
        return self.class_to_id.get(resolved, 0)


@dataclass
class AnatomyClassConfig:
    """Class labels for the anatomy segmentation task (separate from instruments).

    The anatomy annotators labeled both instruments visible in the same frames
    and anatomical structures. We keep them all as classes here so the model
    learns to distinguish anatomy from the tools that occlude it.
    """
    class_to_id: Dict[str, int] = field(default_factory=lambda: {
        "Background": 0,
        "BOT": 1,                  # Base of Tongue (anatomy)
        "Soft Palate": 2,          # anatomy
        "Pharyngeal": 3,           # anatomy
        "Pterygoid": 4,            # anatomy
        "Fat": 5,                  # anatomy (Fascia annotations were merged here)
        "Cut": 6,                  # incision/cut region (treated as a region class)
        "Bipolar": 7,              # instrument
        "Monopolar": 8,            # instrument
        "Suction": 9,              # instrument
        "Endotracheal Tube": 10,   # instrument
        "Ligasure": 11,            # instrument
    })

    aliases: Dict[str, str] = field(default_factory=lambda: {
        # Case variations seen in the wild
        "bipolar": "Bipolar",
        "monopolar": "Monopolar",
        "suction": "Suction",
        "soft palate": "Soft Palate",
        "Soft palate": "Soft Palate",
        "pterygoid": "Pterygoid",
        "endotracheal tube": "Endotracheal Tube",
        # Anatomy duplicates
        "Fascia": "Fat",
        "Fat\nFascia": "Fat",
        "fat": "Fat",
        "fascia": "Fat",
        "BOT ": "BOT",
        "bot": "BOT",
        # Some files use a literal "default" placeholder for unset
        "default": "Background",
    })

    # Multipliers for visually-difficult or rare classes — populated empirically
    # after a first training run shows which classes collapse to background.
    weight_multipliers: Dict[str, float] = field(default_factory=lambda: {})

    background_weight: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def id_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}

    def resolve_class_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def get_class_id(self, name: str) -> int:
        resolved = self.resolve_class_name(name)
        return self.class_to_id.get(resolved, 0)

    @property
    def class_weights(self) -> List[float]:
        # Fallback weights (uniform 1.0 for non-bg). Real weights are computed
        # live from masks via compute_class_weights_from_masks().
        return [self.background_weight] + [1.0] * (self.num_classes - 1)


@dataclass
class JML619ClassConfig:
    """Class labels for the JML619 case (VIA video project, free-text labels).

    Canonical names follow the team's class_merge_default.json convention
    (lowercase). The merge map itself is applied at parse time; `aliases` here
    only covers spellings the merge map doesn't know about (typos, pre-merge
    synonyms). Anatomy classes first, instruments after.

    palatoglossus/palatopharyngeus muscle and ligasure have 1/1/4 polygons in
    the dataset — kept so no annotation is dropped, but expect no signal;
    the weight clamp in compute_class_weights_from_masks keeps them from
    destabilizing training.
    """
    class_to_id: Dict[str, int] = field(default_factory=lambda: {
        "background": 0,
        "base of tongue": 1,
        "soft palate": 2,
        "uvula": 3,
        "posterior pharyngeal wall": 4,
        "superior constrictor muscle": 5,
        "medial pterygoid muscle": 6,
        "parapharyngeal fat": 7,
        "prevertebral fascia": 8,
        "palatoglossus muscle": 9,
        "palatopharyngeus muscle": 10,
        "maryland": 11,             # instrument (Maryland dissector)
        "monopolar": 12,            # instrument
        "suction": 13,              # instrument
        "ligasure": 14,             # instrument
    })

    # Applied to lowercased labels BEFORE the class_merge_default.json map.
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "monopolar cauteryy": "monopolar cautery",  # annotation typo
        "_default": "background",
    })

    weight_multipliers: Dict[str, float] = field(default_factory=lambda: {})

    background_weight: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def id_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}

    def resolve_class_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def get_class_id(self, name: str) -> int:
        resolved = self.resolve_class_name(name)
        return self.class_to_id.get(resolved, 0)

    @property
    def class_weights(self) -> List[float]:
        return [self.background_weight] + [1.0] * (self.num_classes - 1)


@dataclass
class UnifiedClassConfig:
    """Unified label space for combined JSR + JML619 training.

    Decisions and rationale are documented in data/unified_schema.md (D1-D6).
    Key defaults: JSR "Bipolar" merges into "maryland" (same instrument, D1);
    the three fat/fascia vocabularies stay separate (D2); JSR-only classes
    "cut" and "endotracheal tube" are kept (D3).
    """
    class_to_id: Dict[str, int] = field(default_factory=lambda: {
        "background": 0,
        "base of tongue": 1,
        "soft palate": 2,
        "uvula": 3,
        "posterior pharyngeal wall": 4,
        "superior constrictor muscle": 5,
        "medial pterygoid muscle": 6,
        "parapharyngeal fat": 7,
        "prevertebral fascia": 8,
        "fat fascia": 9,               # JSR coarse fat/fascia (D2)
        "palatoglossus muscle": 10,
        "palatopharyngeus muscle": 11,
        "cut": 12,                     # JSR only (D3)
        "maryland": 13,                # includes JSR "Bipolar" (D1)
        "monopolar": 14,
        "suction": 15,
        "endotracheal tube": 16,       # JSR only (D3)
        "ligasure": 17,
    })

    # Applied to lowercased labels BEFORE class_merge_default.json. Covers the
    # JML typo/placeholder plus the JSR->unified renames the merge map lacks.
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "monopolar cauteryy": "monopolar cautery",
        "_default": "background",
        "bipolar": "maryland",                       # D1
        "bot": "base of tongue",
        "pharyngeal": "posterior pharyngeal wall",   # D5
    })

    weight_multipliers: Dict[str, float] = field(default_factory=lambda: {})

    background_weight: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def id_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}

    def resolve_class_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def get_class_id(self, name: str) -> int:
        resolved = self.resolve_class_name(name)
        return self.class_to_id.get(resolved, 0)

    @property
    def class_weights(self) -> List[float]:
        return [self.background_weight] + [1.0] * (self.num_classes - 1)


@dataclass
class DataConfig:
    """Data processing configuration."""
    image_size: Tuple[int, int] = (512, 512)
    train_split: float = 0.8
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 7
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Early stopping
    early_stopping_patience: int = 15

    # Gradient clipping for MPS stability
    max_grad_norm: float = 1.0

    # Loss function. Tried gamma=2.5 to escape Retractor's background-collapse
    # but it broke Needle Driver (whose v2 Dice was already borderline at 0.56)
    # by deprioritizing its mostly-correct pixels. Retractor was rescued instead
    # by its weight bump (1.14 -> 3.0); gamma stays at 2.0.
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    focal_weight: float = 0.5

    # DataLoader settings for MPS
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
