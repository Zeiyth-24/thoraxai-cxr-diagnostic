"""Weakly-supervised Grad-CAM localization for DenseNet-121."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetGradCAM:
    """Compute Grad-CAM activation maps without bounding-box supervision."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        # Hook denseblock4 directly rather than norm5 because DenseNet forward applies
        # F.relu(features, inplace=True) right after norm5, which conflicts with backward hooks.
        self.target_layer = (
            target_layer
            if target_layer is not None
            else model.features.denseblock4
        )

    def compute_heatmap_and_probabilities(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Run one forward/backward pass and return (heatmap, sigmoid_probs, target_idx)."""
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected input_tensor shape (1, C, H, W), got {tuple(input_tensor.shape)}."
            )

        captured_activations: list[torch.Tensor] = []
        captured_gradients: list[torch.Tensor] = []

        def forward_hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured_activations.append(output)

        def backward_hook(
            _module: nn.Module,
            _grad_inputs: tuple[torch.Tensor | None, ...],
            grad_outputs: tuple[torch.Tensor | None, ...],
        ) -> None:
            if grad_outputs[0] is not None:
                captured_gradients.append(grad_outputs[0])

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.eval()
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(input_tensor)
                probabilities = (
                    torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
                )

                resolved_idx = (
                    int(np.argmax(probabilities))
                    if target_class_idx is None
                    else int(target_class_idx)
                )
                if not 0 <= resolved_idx < logits.shape[1]:
                    raise ValueError(
                        f"target_class_idx {resolved_idx} out of range for {logits.shape[1]} classes."
                    )

                logits[0, resolved_idx].backward()

            if not captured_activations or not captured_gradients:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations or gradients."
                )

            activations = captured_activations[0].detach()
            gradients = captured_gradients[0].detach()

            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            weighted_activation = (channel_weights * activations).sum(dim=1).squeeze(0)
            heatmap_tensor = F.relu(weighted_activation)
            heatmap = heatmap_tensor.cpu().numpy().astype(np.float32)

            max_val = float(heatmap.max())
            min_val = float(heatmap.min())
            if max_val - min_val > 1e-8:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            return heatmap, probabilities, resolved_idx
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int,
    ) -> np.ndarray:
        """Generate a normalized [0, 1] Grad-CAM heatmap for a specific class index."""
        heatmap, _, _ = self.compute_heatmap_and_probabilities(
            input_tensor=input_tensor,
            target_class_idx=target_class_idx,
        )
        return heatmap


def create_heatmap_overlay(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a [0, 1] Grad-CAM heatmap onto an RGB or grayscale uint8 radiograph."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")

    if base_image.ndim == 2:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.ndim == 3 and base_image.shape[2] == 3:
        base_rgb = base_image
    else:
        raise ValueError(
            f"Expected base_image shape (H, W) or (H, W, 3), got {base_image.shape}."
        )

    height, width = base_rgb.shape[:2]
    resized_heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(255.0 * resized_heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        base_rgb.astype(np.uint8),
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0,
    )
    return blended
