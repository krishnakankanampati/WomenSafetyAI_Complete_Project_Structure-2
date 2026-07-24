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
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ai_model.config import (
    BEST_MODEL_PATH,
    DEVICE,
    ENABLE_LLM_FALLBACK,
    ID_MAP,
    LABEL_MAP,
    LLM_ESCALATION_THRESHOLD,
    MAX_LENGTH,
)
from ai_model.llm_fallback import classify_with_llm
from ai_model.model import load_model
from ai_model.tokenizer import get_tokenizer, tokenize_batch

# Predictions can contain non-ASCII text (Telugu/Tamil/Kannada/etc). stdout's
# default encoding on Windows is the console codepage (e.g. cp1252), not
# UTF-8, which raises UnicodeEncodeError as soon as it's redirected to a
# file/pipe instead of a real console. Force UTF-8 so multilingual output
# never depends on the caller's terminal codepage.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    # "local" or "llm_escalated" - see WomenSafetyPredictor.predict_batch.
    # confidence/probabilities always stay the local model's numbers even
    # when escalated, since that's *why* it escalated; llm_reasoning is only
    # set when a second opinion actually ran.
    source: str = "local"
    llm_reasoning: Optional[str] = None

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

    def __init__(
        self,
        checkpoint_path=BEST_MODEL_PATH,
        device=DEVICE,
        max_length=MAX_LENGTH,
        escalate_low_confidence=ENABLE_LLM_FALLBACK,
    ):
        self.device = device
        self.max_length = max_length
        self.escalate_low_confidence = escalate_low_confidence

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
            confidence = float(probs[label_id])
            prediction = Prediction(
                text=text,
                label=LABEL_MAP[label_id],
                label_id=label_id,
                confidence=confidence,
                probabilities={
                    LABEL_MAP[i]: float(p) for i, p in enumerate(probs)
                },
            )

            if self.escalate_low_confidence and confidence < LLM_ESCALATION_THRESHOLD:
                verdict = classify_with_llm(text)
                if verdict is not None:
                    prediction.label = verdict.label
                    prediction.label_id = ID_MAP[verdict.label]
                    prediction.source = "llm_escalated"
                    prediction.llm_reasoning = verdict.reasoning
                    logger.info(
                        "Escalated (local conf=%.3f) -> %s: %s",
                        confidence, verdict.label, verdict.reasoning,
                    )

            results.append(prediction)
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
    parser.add_argument(
        "--no-llm-fallback", action="store_true",
        help="Disable escalating low-confidence predictions to Gemini (requires GEMINI_API_KEY otherwise)",
    )
    return parser


def _print_result(result: Prediction, as_json: bool):
    if as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False))
    else:
        marker = " [LLM]" if result.source == "llm_escalated" else ""
        print(f"[{result.label:<20}]{marker} confidence={result.confidence:.3f} | {result.text}")


def main():
    args = _build_arg_parser().parse_args()

    predictor = WomenSafetyPredictor(
        checkpoint_path=args.checkpoint,
        escalate_low_confidence=not args.no_llm_fallback,
    )

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
