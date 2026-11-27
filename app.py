"""Gradio web interface for ThoraxAI multi-label chest X-ray diagnosis."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
from PIL import Image

from src.config import (
    AUTO_TARGET_OPTION,
    DEFAULT_MODEL_PATH,
    DISEASE_LABELS,
    MODEL_PATH_ENV_VAR,
    POSITIVE_THRESHOLD,
)
from src.inference import predict_chest_xray
from src.model import load_thoraxai_model, resolve_checkpoint_path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Configurable checkpoint path (defaults to weights/thoraxai_model.pth or THORAXAI_MODEL_PATH).
MODEL_PATH: Path = resolve_checkpoint_path()
EXAMPLES_DIR: Path = Path(__file__).resolve().parent / "examples"

SAMPLE_RADIOGRAPHS: list[list[object]] = [
    [str(EXAMPLES_DIR / path_name), AUTO_TARGET_OPTION, True]
    for path_name in (
        "sample_cardiomegaly.png",
        "sample_effusion.png",
        "sample_nodule.png",
        "sample_pneumothorax.png",
    )
    if (EXAMPLES_DIR / path_name).is_file()
]

CUSTOM_CSS: str = """
.thoraxai-disclaimer {
    background-color: #fff8e6;
    border: 1px solid #d97706;
    border-left: 5px solid #b45309;
    border-radius: 6px;
    padding: 12px 16px;
    color: #1f2937;
    margin-bottom: 12px;
}
.thoraxai-disclaimer strong {
    color: #92400e;
}
.thoraxai-warning {
    background-color: #fef2f2;
    border: 1px solid #dc2626;
    border-left: 5px solid #b91c1c;
    border-radius: 6px;
    padding: 12px 16px;
    color: #1f2937;
    margin-bottom: 12px;
}
"""


def _initialize_model(
    checkpoint_path: Path,
) -> tuple[object | None, object | None, str | None]:
    """Load the DenseNet-121 checkpoint if present, returning a clear error if missing."""
    if not checkpoint_path.is_file():
        message = (
            f"Trained model weights were not found at `{checkpoint_path}`. "
            f"Place `thoraxai_model.pth` at `{DEFAULT_MODEL_PATH}` (or set `{MODEL_PATH_ENV_VAR}`) "
            "and restart the application to enable live inference."
        )
        logger.warning(message)
        return None, None, message

    try:
        model, device = load_thoraxai_model(
            checkpoint_path=checkpoint_path,
            prefer_cuda=False,
        )
        return model, device, None
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"Failed to load checkpoint at `{checkpoint_path}`: {exc}"
        logger.error(message)
        return None, None, message


LOADED_MODEL, LOADED_DEVICE, LOAD_ERROR = _initialize_model(MODEL_PATH)


def _format_probability_rows(probabilities: dict[str, float]) -> list[list[object]]:
    """Build ranked rows for the 14-pathology table tab."""
    rows: list[list[object]] = []
    for rank, (pathology, score) in enumerate(probabilities.items(), start=1):
        status = "Positive (>= 50%)" if score >= POSITIVE_THRESHOLD else "Below Threshold"
        rows.append([rank, pathology, round(score * 100.0, 2), status])
    return rows


def run_diagnostic_analysis(
    image: Image.Image | None,
    target_pathology: str,
    apply_clahe: bool,
) -> tuple[Image.Image | None, dict[str, float], list[list[object]], str]:
    """Execute 14-pathology prediction and Grad-CAM overlay for the Gradio UI."""
    if LOADED_MODEL is None:
        guidance = (
            f"### Model Weights Unavailable\n\n{LOAD_ERROR}"
        )
        return None, {}, [], guidance

    if image is None:
        return (
            None,
            {},
            [],
            "### Waiting for Input\n\nUpload a frontal chest radiograph or select a sample below, then click **Run Diagnostic Analysis**.",
        )

    prediction = predict_chest_xray(
        image=image,
        target_pathology=target_pathology,
        apply_clahe=apply_clahe,
        model=LOADED_MODEL,
        device=LOADED_DEVICE,
    )

    probabilities = prediction.probabilities
    top_pathology, top_score = next(iter(probabilities.items()))
    visualized_pathology = (
        top_pathology
        if not target_pathology or target_pathology == AUTO_TARGET_OPTION
        else target_pathology
    )
    visualized_score = probabilities[visualized_pathology]

    positive_findings = [
        f"- **{label}**: {score * 100:.1f}%"
        for label, score in probabilities.items()
        if score >= POSITIVE_THRESHOLD
    ]

    if positive_findings:
        findings_block = "\n".join(positive_findings)
        summary_markdown = (
            f"### Active Findings (Threshold >= 50%)\n"
            f"{findings_block}\n\n"
            f"**Grad-CAM Target:** `{visualized_pathology}` ({visualized_score * 100:.1f}% confidence) | "
            f"**Visual CLAHE:** `{'Enabled' if apply_clahe else 'Disabled'}`"
        )
    else:
        summary_markdown = (
            "### No Pathology Above 50% Threshold\n"
            f"- Highest predicted probability: **{top_pathology}** ({top_score * 100:.1f}%)\n\n"
            f"**Grad-CAM Target:** `{visualized_pathology}` ({visualized_score * 100:.1f}% confidence) | "
            f"**Visual CLAHE:** `{'Enabled' if apply_clahe else 'Disabled'}`"
        )

    ranked_rows = _format_probability_rows(probabilities)
    return prediction.overlay_image, probabilities, ranked_rows, summary_markdown


def build_demo() -> gr.Blocks:
    """Construct the ThoraxAI clinical Gradio Blocks interface."""
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="ThoraxAI: Multi-Label Chest X-Ray Diagnostic Engine",
    ) as interface:
        gr.Markdown(
            "# ThoraxAI: Multi-Label Chest X-Ray Diagnostic Engine\n"
            "Automated screening across **14 NIH ChestX-ray14 pathologies** powered by "
            "**DenseNet-121**, **weakly-supervised Grad-CAM** (`denseblock4`), and optional **CLAHE** visual contrast enhancement."
        )

        gr.HTML(
            "<div class='thoraxai-disclaimer'>"
            "<strong>Medical Research Disclaimer:</strong> ThoraxAI is an academic and engineering portfolio "
            "Computer-Aided Diagnosis (CAD) prototype. It is <strong>not</strong> a certified medical device and "
            "must <strong>not</strong> be used as a substitute for clinical diagnosis by a licensed radiologist or physician."
            "</div>"
        )

        if LOAD_ERROR is not None:
            gr.HTML(
                "<div class='thoraxai-warning'>"
                f"<strong>Checkpoint Missing:</strong> {LOAD_ERROR}"
                "</div>"
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_radiograph = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray (Frontal PA/AP)",
                )
                target_dropdown = gr.Dropdown(
                    choices=[AUTO_TARGET_OPTION, *DISEASE_LABELS],
                    value=AUTO_TARGET_OPTION,
                    label="Grad-CAM Target Pathology",
                )
                clahe_checkbox = gr.Checkbox(
                    value=True,
                    label="Apply CLAHE Visual Contrast Enhancement",
                    info=(
                        "Enhances vascular and bone contrast on the Grad-CAM overlay "
                        "(clipLimit=2.0, tileGridSize=8x8) while keeping model inference inputs aligned with training."
                    ),
                )
                analyze_button = gr.Button(
                    "Run Diagnostic Analysis",
                    variant="primary",
                )

            with gr.Column(scale=1):
                summary_output = gr.Markdown(
                    value=(
                        "### Ready for Analysis\n\n"
                        "Upload a frontal chest radiograph or choose a sample below, then click **Run Diagnostic Analysis**."
                    )
                )
                with gr.Tabs():
                    with gr.Tab("Grad-CAM & Ranked Confidence"):
                        with gr.Row():
                            gradcam_output = gr.Image(
                                type="pil",
                                label="Grad-CAM Localization Heatmap Overlay",
                            )
                            confidence_output = gr.Label(
                                num_top_classes=14,
                                label="14-Pathology Confidence Scores",
                            )
                    with gr.Tab("Detailed Probability Table"):
                        table_output = gr.Dataframe(
                            headers=[
                                "Rank",
                                "Pathology",
                                "Confidence (%)",
                                "Threshold (>= 50%)",
                            ],
                            datatype=["number", "str", "number", "str"],
                            interactive=False,
                            label="All 14 Pathologies Ranked by Confidence",
                        )

        if SAMPLE_RADIOGRAPHS:
            gr.Examples(
                examples=SAMPLE_RADIOGRAPHS,
                inputs=[input_radiograph, target_dropdown, clahe_checkbox],
                label="Sample NIH ChestX-ray14 Radiographs (Click to Load)",
            )

        analyze_button.click(
            fn=run_diagnostic_analysis,
            inputs=[input_radiograph, target_dropdown, clahe_checkbox],
            outputs=[
                gradcam_output,
                confidence_output,
                table_output,
                summary_output,
            ],
        )

    return interface


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
