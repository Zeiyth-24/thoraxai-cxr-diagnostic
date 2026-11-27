"""ThoraxAI modular inference, model loading, and Grad-CAM package."""

from __future__ import annotations

from src.config import (
    AUTO_TARGET_OPTION,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DEFAULT_MODEL_PATH,
    DISEASE_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_RESOLUTION,
    NUM_CLASSES,
    POSITIVE_THRESHOLD,
)
from src.gradcam import DenseNetGradCAM, create_heatmap_overlay
from src.inference import DiagnosticOutput, predict_chest_xray
from src.model import build_densenet121, load_thoraxai_model, resolve_checkpoint_path

__all__ = [
    "AUTO_TARGET_OPTION",
    "CLAHE_CLIP_LIMIT",
    "CLAHE_TILE_GRID_SIZE",
    "DEFAULT_MODEL_PATH",
    "DISEASE_LABELS",
    "DiagnosticOutput",
    "DenseNetGradCAM",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_RESOLUTION",
    "NUM_CLASSES",
    "POSITIVE_THRESHOLD",
    "build_densenet121",
    "create_heatmap_overlay",
    "load_thoraxai_model",
    "predict_chest_xray",
    "resolve_checkpoint_path",
]
