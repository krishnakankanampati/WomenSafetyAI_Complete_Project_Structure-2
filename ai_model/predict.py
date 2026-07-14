"""
====================================================
Women Safety AI Project
Inference Script
File : predict.py

Loads a trained checkpoint and classifies free-text comments into one
of the categories defined in config.CLASS_NAMES.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.predict --text "some comment"
    python -m ai_model.predict --file comments.txt --json
    python -m ai_model.predict            # interactive mode

Programmatic usage (e.g. from backend/main.py):
    from ai_model.predict import get_predictor
    predictor = get_predictor()
    result = predictor.predict("some comment")
====================================================
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from ai_model.config import BEST_MODEL_PATH, DEVICE, LABEL_MAP, MAX_LENGTH
from ai_model.model import load_model
from ai_model.tokenizer import get_tokenizer, tokenize_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================
# RESULT OBJECT
# ==========================================================

@dataclass
class Prediction:
    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

    def as_dict(self):
        return asdict(self)


# ==========================================================
# PREDICTOR
# ==========================================================

class WomenSafetyPredictor:
    """
    Wraps the trained model + tokenizer for inference.

    Instantiate once and reuse across requests (loading the transformer
    backbone is expensive) - e.g. create a single instance at FastAPI
    startup rather than per-request.
    """

    def __init__(self, checkpoint_path=BEST_MODEL_PATH, device=DEVICE, max_length=MAX_LENGTH):
        self.device = device
        self.max_length = max_length

        logger.info("Loading tokenizer...")
        self.tokenizer = get_tokenizer()

        logger.info("Loading model checkpoint: %s", checkpoint_path)
        self.model = load_model(checkpoint_path, device)
        self.model.eval()

        logger.info("Predictor ready on device=%s", device)

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[Prediction]:
        if not texts:
            return []

        encoding = tokenize_batch(self.tokenizer, list(texts), self.max_length)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs["logits"], dim=1).cpu()

        results = []
        for text, probs in zip(texts, probabilities):
            label_id = int(torch.argmax(probs).item())
            results.append(
                Prediction(
                    text=text,
                    label=LABEL_MAP[label_id],
                    label_id=label_id,
                    confidence=float(probs[label_id]),
                    probabilities={
                        LABEL_MAP[i]: float(p) for i, p in enumerate(probs)
                    },
                )
            )
        return results

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]


# ==========================================================
# SINGLETON ACCESSOR (for reuse by the backend service)
# ==========================================================

_predictor_instance = None


def get_predictor(checkpoint_path=BEST_MODEL_PATH) -> WomenSafetyPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = WomenSafetyPredictor(checkpoint_path=checkpoint_path)
    return _predictor_instance


# ==========================================================
# CLI ENTRY POINT
# ==========================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Women Safety AI - inference CLI")
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file, one sample per line")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_PATH, help="Path to a model checkpoint")
    parser.add_argument("--json", action="store_true", help="Print results as JSON lines")
    return parser


def _print_result(result: Prediction, as_json: bool):
    if as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False))
    else:
        print(f"[{result.label:<20}] confidence={result.confidence:.3f} | {result.text}")


def main():
    args = _build_arg_parser().parse_args()

    predictor = WomenSafetyPredictor(checkpoint_path=args.checkpoint)

    if args.text:
        for result in predictor.predict_batch([args.text]):
            _print_result(result, args.json)
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        for result in predictor.predict_batch(texts):
            _print_result(result, args.json)
        return

    print("Interactive mode. Type a comment and press Enter (Ctrl+C to exit).")
    try:
        while True:
            line = input(">> ").strip()
            if not line:
                continue
            _print_result(predictor.predict(line), args.json)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
