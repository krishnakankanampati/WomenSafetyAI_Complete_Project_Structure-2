import os
import logging
import torch
import torch.nn as nn
from transformers import AutoModel

from ai_model.config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT,
    LABEL_SMOOTHING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class WomenSafetyModel(nn.Module):

    def __init__(self, class_weights=None):

        super().__init__()

        logger.info("Loading Backbone Model...")

        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(DROPOUT)

        # A single linear layer on top of a pooled sentence representation
        # is the standard "linear probe" head for fine-tuning transformer
        # encoders - it overfits less than a deeper head while the encoder
        # itself already does the representation learning.
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

        # class_weights compensates for classes that keep their natural,
        # unbalanced size in train.csv (see dataset/balance_dataset.py) -
        # oversampling/undersampling every class to one flat count was
        # discarding most of the real "Safe" and "Hate Speech" data.
        self.loss_function = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        logger.info("Model Loaded Successfully")

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        """
        Attention-mask-weighted mean over token embeddings.

        RoBERTa-family encoders are not pretrained with a
        next-sentence/CLS-style objective, so mean pooling over real
        (non-padding) tokens is a stronger sentence representation for
        downstream classification than taking the [CLS]/<s> token alone.
        """

        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)

        summed = (last_hidden_state * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = self.mean_pool(
            outputs.last_hidden_state,
            attention_mask
        )

        pooled_output = self.dropout(
            pooled_output
        )

        logits = self.classifier(
            pooled_output
        )

        loss = None

        if labels is not None:

            loss = self.loss_function(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }

    def predict(
        self,
        input_ids,
        attention_mask
    ):

        self.eval()

        with torch.no_grad():

            outputs = self.forward(
                input_ids,
                attention_mask
            )

            prediction = torch.argmax(
                outputs["logits"],
                dim=1
            )

        return prediction


# ============================================================
# MODEL UTILITIES
# ============================================================

def initialize_classifier(model):
    """
    Xavier-initialize the classifier head. The backbone arrives
    pretrained; only this head is freshly initialized.
    """

    torch.nn.init.xavier_uniform_(model.classifier.weight)
    torch.nn.init.zeros_(model.classifier.bias)


def freeze_embeddings(model):
    """
    Freeze embedding layer only.
    """

    for p in model.backbone.embeddings.parameters():
        p.requires_grad = False


def unfreeze_embeddings(model):
    """
    Unfreeze embedding layer.
    """

    for p in model.backbone.embeddings.parameters():
        p.requires_grad = True


def build_layerwise_param_groups(model, base_lr, head_lr, weight_decay, layer_decay=0.85):
    """
    Discriminative (layer-wise-decayed) learning rates: bottom encoder
    layers - which encode general multilingual syntax shared across
    languages/tasks - get a much smaller LR than top layers, which
    encode task-specific semantics and need more room to adapt.

    Tried hard-freezing the bottom 6 of 12 layers first (halving
    trainable params) to fight overfitting on the ~32K-row train set,
    but that just capped the achievable peak lower (val F1 0.80 vs 0.82)
    without fixing the overfit-after-peak pattern - the model had too
    little capacity to fit well, not too much. Layer-wise LR decay
    keeps every layer trainable (so peak capacity isn't lost) while
    still damping updates to the layers most likely to overfit.
    """

    num_layers = len(model.backbone.encoder.layer)

    groups = [
        {
            "params": model.backbone.embeddings.parameters(),
            "lr": base_lr * (layer_decay ** num_layers),
            "weight_decay": weight_decay,
        },
    ]

    for i, layer in enumerate(model.backbone.encoder.layer):
        lr = base_lr * (layer_decay ** (num_layers - 1 - i))
        groups.append({
            "params": layer.parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        })

    groups.append({
        "params": model.classifier.parameters(),
        "lr": head_lr,
        "weight_decay": weight_decay,
    })

    return groups


def print_model_summary(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print("=" * 60)
    print("Women Safety AI Model Summary")
    print("=" * 60)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print(f"Frozen Parameters    : {frozen:,}")
    print("=" * 60)


def create_model(device, class_weights=None):

    model = WomenSafetyModel(class_weights=class_weights)

    initialize_classifier(model)

    model.to(device)

    return model


def save_model(model, path):

    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        path
    )


def load_model(path, device):

    model = WomenSafetyModel()

    checkpoint = torch.load(
        path,
        map_location=device
    )

    # strict=False: checkpoints trained with class-weighted loss carry an
    # extra "loss_function.weight" buffer that older checkpoints don't
    # have, and vice versa. That buffer only affects loss computation
    # during training, never the forward pass, so it's safe to ignore
    # for inference/evaluation here.
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    ignorable = {"loss_function.weight"}
    real_missing = [k for k in missing if k not in ignorable]
    real_unexpected = [k for k in unexpected if k not in ignorable]
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch - missing={real_missing}, unexpected={real_unexpected}"
        )

    model.to(device)

    model.eval()

    return model