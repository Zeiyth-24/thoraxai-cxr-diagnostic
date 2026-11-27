"""Configuration constants for the ThoraxAI diagnostic engine."""

from __future__ import annotations

from pathlib import Path

# Exact training index order from NIH ChestX-ray14 multi-label encoding.
DISEASE_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

NUM_CLASSES: int = len(DISEASE_LABELS)

# ImageNet normalization constants used during DenseNet-121 training.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
INPUT_RESOLUTION: tuple[int, int] = (224, 224)

# CLAHE contrast enhancement settings for visual inspection.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

POSITIVE_THRESHOLD: float = 0.5
AUTO_TARGET_OPTION: str = "Auto (Top Predicted)"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH: Path = REPO_ROOT / "weights" / "thoraxai_model.pth"
FALLBACK_MODEL_PATH: Path = REPO_ROOT / "thoraxai_model.pth"
MODEL_PATH_ENV_VAR: str = "THORAXAI_MODEL_PATH"
