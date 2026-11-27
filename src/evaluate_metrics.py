"""Compute per-class ROC-AUC and PR-AUC on the patient-aware validation split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

# Allow running both `python -m src.evaluate_metrics` and `python src/evaluate_metrics.py`.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DISEASE_LABELS
from src.inference import build_inference_transform
from src.model import load_thoraxai_model

logger = logging.getLogger(__name__)


class PathologyMetrics(NamedTuple):
    """Per-class validation metrics for a single thoracic pathology."""

    pathology: str
    positives: int
    prevalence_pct: float
    roc_auc: float | None
    pr_auc: float | None


class ValidationChestXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset matching the notebook validation transform pipeline."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.image_paths = dataframe["image_path"].tolist()
        self.labels = dataframe[list(DISEASE_LABELS)].to_numpy(dtype=np.float32)
        self.transform = build_inference_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_paths[index]) as img:
            rgb_image = img.convert("RGB")
            tensor = self.transform(rgb_image)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.float32)
        return tensor, label_tensor


def load_patient_aware_validation_split(dataset_dir: Path) -> pd.DataFrame:
    """Reproduce the exact 80/20 GroupShuffleSplit(random_state=42) validation split."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"NIH ChestX-ray14 dataset directory not found: '{dataset_dir}'."
        )

    csv_candidates = sorted(dataset_dir.rglob("Data_Entry_2017*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"Could not find Data_Entry_2017*.csv inside '{dataset_dir}'."
        )
    metadata_path = csv_candidates[0]
    metadata_df = pd.read_csv(metadata_path)

    image_index_map = {
        png_path.name: str(png_path) for png_path in dataset_dir.rglob("*.png")
    }
    metadata_df["image_path"] = metadata_df["Image Index"].map(image_index_map)
    metadata_df = metadata_df.dropna(subset=["image_path"]).reset_index(drop=True)

    for disease in DISEASE_LABELS:
        metadata_df[disease] = metadata_df["Finding Labels"].apply(
            lambda labels: 1 if disease in str(labels) else 0
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(
        splitter.split(metadata_df, groups=metadata_df["Patient ID"])
    )

    train_patients = set(metadata_df.iloc[train_indices]["Patient ID"].unique())
    val_df = metadata_df.iloc[val_indices].reset_index(drop=True)
    val_patients = set(val_df["Patient ID"].unique())

    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} overlapping Patient IDs."
        )

    logger.info(
        "Validation split ready: %d images across %d unique patients (0 overlap).",
        len(val_df),
        len(val_patients),
    )
    return val_df


def compute_per_class_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[PathologyMetrics], float | None, float | None]:
    """Compute per-class ROC-AUC, PR-AUC, and macro averages across 14 pathologies."""
    rows: list[PathologyMetrics] = []
    valid_roc_aucs: list[float] = []
    valid_pr_aucs: list[float] = []
    sample_count = len(targets)

    for class_idx, pathology in enumerate(DISEASE_LABELS):
        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]
        positives = int(np.sum(y_true == 1))
        prevalence_pct = (positives / sample_count * 100.0) if sample_count else 0.0

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            roc_auc: float | None = None
            pr_auc: float | None = None
        else:
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc = float(average_precision_score(y_true, y_score))
            valid_roc_aucs.append(roc_auc)
            valid_pr_aucs.append(pr_auc)

        rows.append(
            PathologyMetrics(
                pathology=pathology,
                positives=positives,
                prevalence_pct=prevalence_pct,
                roc_auc=roc_auc,
                pr_auc=pr_auc,
            )
        )

    macro_roc = float(np.mean(valid_roc_aucs)) if valid_roc_aucs else None
    macro_pr = float(np.mean(valid_pr_aucs)) if valid_pr_aucs else None
    return rows, macro_roc, macro_pr


def format_metrics_markdown(
    pathology_rows: list[PathologyMetrics],
    macro_roc_auc: float | None,
    macro_pr_auc: float | None,
) -> str:
    """Format 14-class evaluation results as a Markdown table ready for README.md."""
    lines = [
        "| Pathology | Validation Positives | Prevalence (%) | ROC-AUC | PR-AUC |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in pathology_rows:
        roc_text = f"{row.roc_auc:.4f}" if row.roc_auc is not None else "N/A"
        pr_text = f"{row.pr_auc:.4f}" if row.pr_auc is not None else "N/A"
        lines.append(
            f"| `{row.pathology}` | {row.positives:,} | {row.prevalence_pct:.2f}% | {roc_text} | {pr_text} |"
        )

    macro_roc_text = f"**{macro_roc_auc:.4f}**" if macro_roc_auc is not None else "N/A"
    macro_pr_text = f"**{macro_pr_auc:.4f}**" if macro_pr_auc is not None else "N/A"
    lines.append(
        f"| **Macro Mean (14 Classes)** | - | - | {macro_roc_text} | {macro_pr_text} |"
    )
    return "\n".join(lines)


def run_validation_inference(
    val_df: pd.DataFrame,
    model_path: Path | None,
    batch_size: int,
    num_workers: int,
    device_preference: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference over the validation dataframe."""
    if device_preference == "auto":
        prefer_cuda = torch.cuda.is_available()
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device cuda, but no GPU is available.")
        prefer_cuda = True
    else:
        prefer_cuda = False

    model, active_device = load_thoraxai_model(
        checkpoint_path=model_path,
        prefer_cuda=prefer_cuda,
    )

    dataset = ValidationChestXrayDataset(val_df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(active_device.type == "cuda"),
    )

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(active_device, non_blocking=(active_device.type == "cuda"))
            if active_device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    return np.vstack(all_targets), np.vstack(all_probs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for per-class validation evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-class ROC-AUC and PR-AUC across all 14 NIH ChestX-ray14 "
            "pathologies on the patient-aware validation split."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("nih_dataset"),
        help="Root directory containing Data_Entry_2017.csv and NIH PNG radiographs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained DenseNet-121 checkpoint (.pth/.pt). Defaults to weights/thoraxai_model.pth.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run evaluation on (default: auto).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to write the generated Markdown table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)

    val_df = load_patient_aware_validation_split(args.dataset_dir)
    targets, probabilities = run_validation_inference(
        val_df=val_df,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_preference=args.device,
    )
    rows, macro_roc, macro_pr = compute_per_class_metrics(targets, probabilities)
    markdown_table = format_metrics_markdown(rows, macro_roc, macro_pr)

    sys.stdout.write(f"\n{markdown_table}\n")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(f"{markdown_table}\n", encoding="utf-8")
        logger.info("Saved evaluation table to %s", args.output_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
