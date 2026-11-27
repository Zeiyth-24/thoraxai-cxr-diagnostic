"""Preprocessing, inference, and Grad-CAM overlay pipeline for ThoraxAI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.config import (
    AUTO_TARGET_OPTION,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DISEASE_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_RESOLUTION,
)
from src.gradcam import DenseNetGradCAM, create_heatmap_overlay
from src.model import load_thoraxai_model, resolve_checkpoint_path

MAX_DISPLAY_EDGE: int = 512


class DiagnosticOutput(NamedTuple):
    """Structured diagnostic prediction and visual explanation."""

    probabilities: dict[str, float]
    overlay_image: Image.Image


def convert_to_rgb_pil(radiograph: Image.Image | np.ndarray) -> Image.Image:
    """Convert a PIL Image or NumPy radiograph array into an 8-bit RGB PIL Image."""
    if isinstance(radiograph, Image.Image):
        return radiograph.convert("RGB")

    if isinstance(radiograph, np.ndarray):
        arr = np.asarray(radiograph)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
            arr_uint8 = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

        if arr_uint8.ndim == 2:
            return Image.fromarray(arr_uint8, mode="L").convert("RGB")
        if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
            return Image.fromarray(arr_uint8, mode="RGB")
        if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 4:
            return Image.fromarray(arr_uint8, mode="RGBA").convert("RGB")

        raise ValueError(
            f"Unsupported NumPy image shape {arr.shape}; expected 2D grayscale or 3/4-channel image."
        )

    raise TypeError(
        f"Expected PIL.Image.Image or numpy.ndarray, got {type(radiograph).__name__}."
    )


def apply_clahe_contrast(rgb_radiograph: Image.Image) -> Image.Image:
    """Apply CLAHE on luminance channel for clearer radiographic structure inspection."""
    grayscale = np.asarray(rgb_radiograph.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    enhanced = clahe.apply(grayscale)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def build_inference_transform() -> transforms.Compose:
    """Build deterministic ImageNet-normalized transform matching validation."""
    return transforms.Compose(
        [
            transforms.Resize(INPUT_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _resize_for_display(rgb_radiograph: Image.Image) -> np.ndarray:
    """Keep display resolution crisp up to MAX_DISPLAY_EDGE without upscaling small inputs."""
    width, height = rgb_radiograph.size
    longest_edge = max(width, height)
    if longest_edge <= MAX_DISPLAY_EDGE:
        return np.asarray(rgb_radiograph, dtype=np.uint8)

    scale = MAX_DISPLAY_EDGE / float(longest_edge)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = rgb_radiograph.resize(resized_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resolve_target_class_index(target_pathology: str | None) -> int | None:
    """Map user selection to a 0..13 index or None for auto top-predicted selection."""
    if target_pathology is None or target_pathology == "" or target_pathology == AUTO_TARGET_OPTION:
        return None
    if target_pathology not in DISEASE_LABELS:
        valid_options = ", ".join(DISEASE_LABELS)
        raise ValueError(
            f"Unknown pathology '{target_pathology}'. Expected '{AUTO_TARGET_OPTION}' or one of: {valid_options}."
        )
    return DISEASE_LABELS.index(target_pathology)


@lru_cache(maxsize=2)
def _get_cached_model(
    checkpoint_path_str: str,
    prefer_cuda: bool = False,
) -> tuple[nn.Module, torch.device]:
    """Cache loaded DenseNet-121 model in memory across repeated predictions."""
    return load_thoraxai_model(
        checkpoint_path=Path(checkpoint_path_str),
        prefer_cuda=prefer_cuda,
    )


def predict_chest_xray(
    image: Image.Image | np.ndarray,
    target_pathology: str | None = None,
    apply_clahe: bool = True,
    *,
    model: nn.Module | None = None,
    device: torch.device | None = None,
    model_path: str | Path | None = None,
) -> DiagnosticOutput:
    """Predict all 14 thoracic pathology probabilities and build a Grad-CAM overlay.

    Model inference always uses standard ImageNet-normalized radiographs to stay
    aligned with training, while optional CLAHE enhances the visual canvas used
    to render the Grad-CAM overlay.
    """
    if image is None:
        raise ValueError("Input chest radiograph cannot be None.")

    target_idx = _resolve_target_class_index(target_pathology)
    raw_rgb = convert_to_rgb_pil(image)

    # Keep model tensor distribution aligned with validation (no CLAHE skew on logits).
    preprocess = build_inference_transform()
    input_tensor = preprocess(raw_rgb).unsqueeze(0)

    if model is not None:
        active_model = model
        active_device = (
            device
            if device is not None
            else next(active_model.parameters()).device
        )
    else:
        resolved_checkpoint = resolve_checkpoint_path(model_path)
        active_model, active_device = _get_cached_model(
            str(resolved_checkpoint),
            prefer_cuda=False,
        )

    input_tensor = input_tensor.to(active_device)
    gradcam = DenseNetGradCAM(active_model)
    heatmap, probabilities, _ = gradcam.compute_heatmap_and_probabilities(
        input_tensor=input_tensor,
        target_class_idx=target_idx,
    )

    sorted_pairs = sorted(
        zip(DISEASE_LABELS, probabilities.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    sorted_probabilities: dict[str, float] = {
        label: float(score) for label, score in sorted_pairs
    }

    visual_canvas = apply_clahe_contrast(raw_rgb) if apply_clahe else raw_rgb
    base_rgb = _resize_for_display(visual_canvas)
    overlay_rgb = create_heatmap_overlay(base_image=base_rgb, heatmap=heatmap)
    overlay_pil = Image.fromarray(overlay_rgb, mode="RGB")

    return DiagnosticOutput(
        probabilities=sorted_probabilities,
        overlay_image=overlay_pil,
    )
