"""
====================================================
Women Safety AI Project
Model Training Script
File : train.py

Fine-tunes the transformer-based classifier (see model.py) on the
train/validation splits produced by dataset.py.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.train
    python -m ai_model.train --epochs 10
    python -m ai_model.train --resume saved_models/last_model.pt
====================================================
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai_model.config import (
    BEST_MODEL_PATH,
    DEVICE,
    EARLY_STOPPING,
    EPOCHS,
    HEAD_LEARNING_RATE,
    LAST_MODEL_PATH,
    LEARNING_RATE,
    MODEL_DIR,
    PATIENCE,
    PRINT_EVERY,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from ai_model.dataset import create_train_loader, create_validation_loader, compute_class_weights
from ai_model.model import (
    build_layerwise_param_groups,
    create_model,
    freeze_embeddings,
    print_model_summary,
    save_model,
    unfreeze_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================
# METRICS
# ==========================================================

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self):
        return asdict(self)


def compute_metrics(labels, predictions, loss):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return EpochMetrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ==========================================================
# EARLY STOPPING
# ==========================================================

class EarlyStopper:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value):
        improved = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ==========================================================
# TRAINER
# ==========================================================

class Trainer:

    def __init__(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, resume_path=None):
        self.epochs = epochs

        logger.info("Creating data loaders...")
        self.train_loader = create_train_loader()
        self.valid_loader = create_validation_loader()

        logger.info("Computing class weights from train.csv distribution...")
        class_weights = compute_class_weights().to(DEVICE)
        logger.info("Class weights: %s", class_weights.tolist())

        logger.info("Creating model...")
        self.model = create_model(DEVICE, class_weights=class_weights)

        # Keep the pretrained multilingual embeddings frozen for epoch 1
        # so the randomly-initialized classifier head doesn't push large,
        # noisy gradients back into them before it has learned anything
        # useful. Unfrozen again after the first epoch (see fit()).
        freeze_embeddings(self.model)

        print_model_summary(self.model)

        logger.info("Creating optimizer and scheduler...")
        # Layer-wise-decayed learning rates across the encoder, plus a much
        # higher LR for the freshly-initialized classifier head - see
        # build_layerwise_param_groups docstring for why this replaced a
        # hard freeze of the bottom 6 layers.
        self.optimizer = AdamW(
            build_layerwise_param_groups(
                self.model,
                base_lr=learning_rate,
                head_lr=HEAD_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                layer_decay=0.85,
            )
        )

        total_steps = max(len(self.train_loader) * epochs, 1)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

        self.best_f1 = 0.0
        self.start_epoch = 0
        self.history = []
        self.early_stopper = EarlyStopper(PATIENCE) if EARLY_STOPPING else None

        if resume_path:
            self._load_checkpoint(resume_path)

            if self.start_epoch > 0:
                # Past the epoch-1 warmup already - resume with everything trainable.
                unfreeze_embeddings(self.model)

        logger.info("Trainer ready. Training on device=%s", DEVICE)

    # ------------------------------------------------------
    # ONE PASS OVER A LOADER (TRAIN OR VALIDATE)
    # ------------------------------------------------------

    def _run_epoch(self, loader, train):
        self.model.train(train)

        total_loss = 0.0
        all_preds, all_labels = [], []

        progress = tqdm(loader, desc="Training" if train else "Validation", leave=False)

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=DEVICE.type, enabled=self.scaler.is_enabled()):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                    logits = outputs["logits"]

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            if train and (step + 1) % PRINT_EVERY == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        return compute_metrics(all_labels, all_preds, avg_loss)

    def train_epoch(self):
        return self._run_epoch(self.train_loader, train=True)

    def validate(self):
        with torch.no_grad():
            return self._run_epoch(self.valid_loader, train=False)

    # ------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------

    def _save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
            },
            LAST_MODEL_PATH,
        )

    def _load_checkpoint(self, path):
        logger.info("Resuming from checkpoint: %s", path)
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_f1 = checkpoint.get("best_f1", 0.0)
        logger.info(
            "Resumed at epoch %d (best_f1=%.4f)", self.start_epoch, self.best_f1
        )

    def _save_history(self):
        history_path = Path(MODEL_DIR) / "training_history.json"
        history_path.write_text(json.dumps(self.history, indent=2))
        logger.info("Training history saved -> %s", history_path)

    # ------------------------------------------------------
    # MAIN TRAINING LOOP
    # ------------------------------------------------------

    def fit(self):
        logger.info("=" * 60)
        logger.info("Training Started")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch()
            valid_metrics = self.validate()

            elapsed = time.time() - t0

            logger.info("-" * 60)
            logger.info("Epoch [%d/%d] (%.1fs)", epoch + 1, self.epochs, elapsed)
            logger.info(
                "Train | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
                train_metrics.loss, train_metrics.accuracy,
                train_metrics.precision, train_metrics.recall, train_metrics.f1,
            )
            logger.info(
                "Valid | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
                valid_metrics.loss, valid_metrics.accuracy,
                valid_metrics.precision, valid_metrics.recall, valid_metrics.f1,
            )

            if epoch == 0:
                unfreeze_embeddings(self.model)
                logger.info("Epoch 1 warmup done - embeddings unfrozen.")

            self.history.append({
                "epoch": epoch + 1,
                "train": train_metrics.as_dict(),
                "valid": valid_metrics.as_dict(),
            })

            self._save_checkpoint(epoch)

            if valid_metrics.f1 > self.best_f1:
                self.best_f1 = valid_metrics.f1
                save_model(self.model, BEST_MODEL_PATH)
                logger.info("New best model saved (val_f1=%.4f) -> %s", self.best_f1, BEST_MODEL_PATH)

            if self.early_stopper and self.early_stopper.step(valid_metrics.f1):
                logger.info(
                    "Early stopping triggered after epoch %d (patience=%d)",
                    epoch + 1, PATIENCE,
                )
                break

        self._save_history()

        logger.info("=" * 60)
        logger.info("Training Completed | Best Validation F1 = %.4f", self.best_f1)
        logger.info("=" * 60)


# ==========================================================
# CLI ENTRY POINT
# ==========================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the Women Safety AI classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    logger.info("=" * 60)
    logger.info("Women Safety AI Training")
    logger.info("=" * 60)

    try:
        trainer = Trainer(
            epochs=args.epochs,
            learning_rate=args.lr,
            resume_path=args.resume,
        )
        trainer.fit()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception:
        logger.exception("Training failed with an unhandled exception.")
        raise


if __name__ == "__main__":
    main()
