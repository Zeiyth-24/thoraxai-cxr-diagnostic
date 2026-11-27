"""DenseNet-121 model architecture and CPU-safe checkpoint loading."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    DEFAULT_MODEL_PATH,
    FALLBACK_MODEL_PATH,
    MODEL_PATH_ENV_VAR,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


def resolve_device(prefer_cuda: bool = False) -> torch.device:
    """Return CPU by default for Hugging Face Spaces CPU Basic compatibility."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    """Resolve checkpoint path from explicit arg, env var, or default locations."""
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser()

    env_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    if FALLBACK_MODEL_PATH.is_file():
        return FALLBACK_MODEL_PATH

    return DEFAULT_MODEL_PATH


def build_densenet121(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
) -> nn.Module:
    """Build DenseNet-121 with the 14-class linear head matching training."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    densenet = models.densenet121(weights=weights)
    in_features = densenet.classifier.in_features
    densenet.classifier = nn.Linear(in_features, num_classes)
    return densenet


def _extract_state_dict(loaded_payload: object) -> dict[str, torch.Tensor]:
    """Normalize raw state_dict or wrapped checkpoint dicts and strip DDP prefixes."""
    if not isinstance(loaded_payload, Mapping):
        raise ValueError(
            "Checkpoint payload must be a state_dict mapping or checkpoint dictionary."
        )

    if "model_state_dict" in loaded_payload and isinstance(
        loaded_payload["model_state_dict"], Mapping
    ):
        raw_state = loaded_payload["model_state_dict"]
    elif "state_dict" in loaded_payload and isinstance(
        loaded_payload["state_dict"], Mapping
    ):
        raw_state = loaded_payload["state_dict"]
    else:
        raw_state = loaded_payload

    cleaned_state: dict[str, torch.Tensor] = {}
    for param_name, param_tensor in raw_state.items():
        clean_name = (
            param_name[7:] if param_name.startswith("module.") else param_name
        )
        cleaned_state[clean_name] = param_tensor
    return cleaned_state


def load_thoraxai_model(
    checkpoint_path: str | Path | None = None,
    prefer_cuda: bool = False,
) -> tuple[nn.Module, torch.device]:
    """Load trained DenseNet-121 weights onto CPU (or CUDA if requested) in eval mode."""
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{resolved_path}'. "
            f"Place your trained weights at '{DEFAULT_MODEL_PATH}' or set "
            f"the '{MODEL_PATH_ENV_VAR}' environment variable."
        )

    target_device = resolve_device(prefer_cuda=prefer_cuda)
    payload = torch.load(resolved_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(payload)

    densenet = build_densenet121(num_classes=NUM_CLASSES, pretrained=False)
    densenet.load_state_dict(state_dict, strict=True)
    densenet.to(target_device)
    densenet.eval()

    logger.info("Loaded ThoraxAI checkpoint from %s onto %s", resolved_path, target_device)
    return densenet, target_device
