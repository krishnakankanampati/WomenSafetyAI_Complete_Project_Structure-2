"""
====================================================
Women Safety AI Project
Model Evaluation Script
File : evaluate.py

Runs a trained checkpoint over the test (or validation) split and
reports accuracy, precision/recall/F1 (weighted + macro), a full
per-class classification report and a confusion matrix. Results are
written to saved_models/reports/.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.evaluate
    python -m ai_model.evaluate --split validation
    python -m ai_model.evaluate --checkpoint saved_models/last_model.pt
====================================================
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ai_model.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    ID_MAP,
    MODEL_DIR,
    TEST_FILE,
    VALID_FILE,
)
from ai_model.dataset import create_dataloader
from ai_model.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(MODEL_DIR) / "reports"


# ==========================================================
# INFERENCE
# ==========================================================

@torch.inference_mode()
def run_inference(model, loader, device):
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs["logits"], dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


# ==========================================================
# METRICS / REPORTING
# ==========================================================

def build_report(labels, preds):
    present_ids = sorted(ID_MAP.values())
    target_names = [CLASS_NAMES[i] for i in present_ids]

    accuracy = accuracy_score(labels, preds)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    per_class_report = classification_report(
        labels, preds,
        labels=present_ids,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        labels, preds,
        labels=present_ids,
        target_names=target_names,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=present_ids)

    metrics = {
        "accuracy": accuracy,
        "weighted": {"precision": precision_w, "recall": recall_w, "f1": f1_w},
        "macro": {"precision": precision_m, "recall": recall_m, "f1": f1_m},
        "per_class": per_class_report,
        "confusion_matrix": cm.tolist(),
        "class_names": target_names,
    }
    return metrics, report_text, cm, target_names


def save_confusion_matrix_plot(cm, class_names, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Women Safety AI - Confusion Matrix")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix plot saved -> %s", output_path)


# ==========================================================
# MAIN EVALUATION ROUTINE
# ==========================================================

def evaluate(checkpoint_path, split_file, split_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model checkpoint: %s", checkpoint_path)
    model = load_model(checkpoint_path, DEVICE)

    logger.info("Loading %s split: %s", split_name, split_file)
    loader = create_dataloader(split_file, shuffle=False)

    labels, preds = run_inference(model, loader, DEVICE)
    metrics, report_text, cm, target_names = build_report(labels, preds)

    logger.info("=" * 60)
    logger.info("Evaluation Results (%s split)", split_name)
    logger.info("=" * 60)
    logger.info("Samples          : %d", len(labels))
    logger.info("Accuracy         : %.4f", metrics["accuracy"])
    logger.info("Weighted F1      : %.4f", metrics["weighted"]["f1"])
    logger.info("Macro F1         : %.4f", metrics["macro"]["f1"])
    logger.info("\n%s", report_text)

    metrics_path = REPORTS_DIR / f"{split_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved -> %s", metrics_path)

    cm_path = REPORTS_DIR / f"{split_name}_confusion_matrix.png"
    save_confusion_matrix_plot(cm, target_names, cm_path)

    return metrics


# ==========================================================
# CLI ENTRY POINT
# ==========================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate the Women Safety AI classifier")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_PATH, help="Path to a model checkpoint")
    parser.add_argument("--split", choices=["test", "validation"], default="test", help="Which split to evaluate on")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    split_file = TEST_FILE if args.split == "test" else VALID_FILE

    try:
        evaluate(args.checkpoint, split_file, args.split)
    except Exception:
        logger.exception("Evaluation failed with an unhandled exception.")
        raise


if __name__ == "__main__":
    main()
