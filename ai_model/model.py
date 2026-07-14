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

    def __init__(self):

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

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

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


def create_model(device):

    model = WomenSafetyModel()

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model