"""
====================================================
Women Safety AI Project
File : verify_llm_fallback.py

Checks whether the Gemini fallback (ai_model/llm_fallback.py) actually
corrects the known misses from this session's held-out evaluations
(saved_models/reports/{big_test,fresh50,simple50}_results.csv).

Runs Gemini against every miss directly - not just the ones below
LLM_ESCALATION_THRESHOLD - so the report also shows which wrong,
high-confidence predictions the threshold-based escalation would never
have caught in normal operation.

Results go to llm_fallback_verification_flash.csv. The earlier
gemini-3.5-flash-lite run over the same 76 misses is preserved as
llm_fallback_verification.csv (40/76 correct); if it's present, the
report prints a head-to-head so a model change can be judged on the same
texts rather than on impressions.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. Get one
from Google AI Studio (aistudio.google.com/apikey) - this script can't
create or supply one for you.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.verify_llm_fallback
====================================================
"""

import logging
import time
from pathlib import Path

import pandas as pd
from google.genai import errors

from ai_model.config import LLM_ESCALATION_THRESHOLD, LLM_FALLBACK_MODEL
from ai_model.llm_fallback import QuotaExhausted, classify_or_raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "reports"
SOURCE_FILES = ["big_test_results.csv", "fresh50_results.csv", "simple50_results.csv"]
OUTPUT_FILE = REPORTS_DIR / "llm_fallback_verification_flash.csv"

# The completed gemini-3.5-flash-lite run (40/76). Preserved rather than
# overwritten - it's the only apples-to-apples baseline for whether moving
# up to the full flash tier actually bought anything.
BASELINE_FILE = REPORTS_DIR / "llm_fallback_verification.csv"

# Free-tier limits are per-minute as well as per-day and Google doesn't
# publish a stable number per model, so rather than hard-coding a guess this
# starts polite and backs off on the actual 429. gemini-3.5-flash also takes
# ~8s per call on its own, which already spaces requests out considerably.
REQUEST_INTERVAL_SECONDS = 4
QUOTA_WAIT_SECONDS = 60
MAX_QUOTA_RETRIES = 3

# 503 UNAVAILABLE ("model is currently experiencing high demand") is a
# server-side capacity blip, not a quota wall - it clears in seconds and is
# common on the popular flash tier. Retried separately from quota with a much
# shorter wait, since there is no daily-reset case to rule out here.
SERVER_ERROR_WAIT_SECONDS = 15
MAX_SERVER_ERROR_RETRIES = 5


def load_misses():
    frames = []
    for name in SOURCE_FILES:
        path = REPORTS_DIR / name
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["source_file"] = name
        frames.append(df[~df["correct"]])
    if not frames:
        raise FileNotFoundError(f"None of {SOURCE_FILES} found under {REPORTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def _classify_with_backoff(text):
    """
    Returns a verdict, or None once retries are exhausted.

    Two distinct transient failures are worth riding out rather than aborting a
    76-row sweep partway:
      429 - quota. Could be the per-minute or the per-day limit; waiting
            distinguishes them cheaply (per-minute clears, daily doesn't).
      503 - the model is busy. Always transient, so retried faster.
    """
    quota_tries = server_tries = 0
    while True:
        try:
            return classify_or_raise(text)

        except QuotaExhausted:
            quota_tries += 1
            if quota_tries >= MAX_QUOTA_RETRIES:
                logger.error("Quota still exhausted after %d retries - likely the "
                             "daily limit. Progress is saved; rerun tomorrow to "
                             "resume.", MAX_QUOTA_RETRIES)
                return None
            logger.warning("Quota hit - waiting %ds before retry %d/%d...",
                           QUOTA_WAIT_SECONDS, quota_tries, MAX_QUOTA_RETRIES - 1)
            time.sleep(QUOTA_WAIT_SECONDS)

        except errors.ServerError as e:
            server_tries += 1
            if server_tries >= MAX_SERVER_ERROR_RETRIES:
                logger.error("Model still unavailable after %d retries: %s",
                             MAX_SERVER_ERROR_RETRIES, str(e)[:160])
                return None
            logger.warning("Model busy (503) - waiting %ds before retry %d/%d...",
                           SERVER_ERROR_WAIT_SECONDS, server_tries,
                           MAX_SERVER_ERROR_RETRIES - 1)
            time.sleep(SERVER_ERROR_WAIT_SECONDS)

        except Exception as e:
            logger.error("Call failed (%s): %s", type(e).__name__, str(e)[:200])
            return None


def _report_baseline_comparison(out):
    """Head-to-head against the preserved flash-lite run, on shared texts only."""
    if not BASELINE_FILE.exists():
        return

    baseline = pd.read_csv(BASELINE_FILE, encoding="utf-8-sig")
    merged = out.merge(
        baseline[["text", "gemini_verdict", "gemini_correct"]], on="text", how="inner"
    )
    if merged.empty:
        return

    new_only = int((merged["llm_correct"] & ~merged["gemini_correct"]).sum())
    old_only = int((~merged["llm_correct"] & merged["gemini_correct"]).sum())

    logger.info("-" * 60)
    logger.info("Head-to-head vs the gemini-3.5-flash-lite run (%d shared texts)", len(merged))
    logger.info("  Both correct                  : %d",
                int((merged["llm_correct"] & merged["gemini_correct"]).sum()))
    logger.info("  Only %-24s: %d", LLM_FALLBACK_MODEL, new_only)
    logger.info("  Only 3.5-flash-lite           : %d", old_only)
    logger.info("  Neither                       : %d",
                int((~merged["llm_correct"] & ~merged["gemini_correct"]).sum()))
    logger.info("  Net change                    : %+d texts", new_only - old_only)


def _report(rows):
    if not rows:
        logger.info("No results collected yet - nothing to report.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    total = len(out)
    fixed = int(out["llm_correct"].sum())
    escalated = out[out["would_auto_escalate"]]
    escalated_fixed = int(escalated["llm_correct"].sum())
    not_escalated = out[~out["would_auto_escalate"]]
    not_escalated_fixed = int(not_escalated["llm_correct"].sum())

    logger.info("=" * 60)
    logger.info("LLM fallback verification results")
    logger.info("Model: %s", LLM_FALLBACK_MODEL)
    logger.info("=" * 60)
    if "llm_model" in out.columns:
        counts = out["llm_model"].value_counts()
        logger.info("Answered by                     : %s",
                    ", ".join("%s x%d" % (m, n) for m, n in counts.items()))
    logger.info("Total known misses tested       : %d", total)
    logger.info("Fixed by the LLM (any)          : %d (%.1f%%)", fixed, 100 * fixed / total)
    logger.info(
        "  Of those below threshold (%.3f) - would auto-escalate in production:",
        LLM_ESCALATION_THRESHOLD,
    )
    logger.info(
        "    %d/%d fixed (%.1f%%)",
        escalated_fixed, len(escalated), 100 * escalated_fixed / max(len(escalated), 1),
    )
    logger.info("  Of those ABOVE threshold - would NOT auto-escalate (confidently wrong):")
    logger.info(
        "    %d/%d fixed (%.1f%%) - only visible by bypassing the threshold, as this script does",
        not_escalated_fixed, len(not_escalated), 100 * not_escalated_fixed / max(len(not_escalated), 1),
    )

    _report_baseline_comparison(out)
    logger.info("Full results saved -> %s", OUTPUT_FILE)


def main():
    misses = load_misses()
    logger.info("Loaded %d known misses across %d files", len(misses), len(SOURCE_FILES))
    logger.info("Model: %s", LLM_FALLBACK_MODEL)

    # The free tier is capped per day - resuming from a prior partial run
    # avoids re-spending that scarce quota on texts already classified.
    rows = []
    already_done = set()
    if OUTPUT_FILE.exists():
        prior = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        rows = prior.to_dict("records")
        already_done = set(prior["text"].astype(str))
        logger.info("Resuming: %d texts already classified in a prior run.", len(already_done))

    pending = misses[~misses["text"].astype(str).isin(already_done)]
    if pending.empty:
        logger.info("Nothing left to classify - all misses already covered.")
        _report(rows)
        return

    logger.info("%d texts remaining to classify.", len(pending))

    first_call = True
    for i, r in pending.iterrows():
        if not first_call:
            time.sleep(REQUEST_INTERVAL_SECONDS)
        first_call = False

        verdict = _classify_with_backoff(str(r["text"]))
        would_escalate = bool(r["confidence"] < LLM_ESCALATION_THRESHOLD)

        if verdict is None:
            logger.error(
                "LLM call failed/unavailable - stopping here (progress saved, rerun to resume)."
            )
            _report(rows)
            return

        llm_correct = verdict.label == r["expected"]
        rows.append({
            "source_file": r["source_file"],
            "language": r["language"],
            "text": r["text"],
            "expected": r["expected"],
            "local_predicted": r["predicted"],
            "local_confidence": round(float(r["confidence"]), 3),
            "would_auto_escalate": would_escalate,
            "llm_verdict": verdict.label,
            "llm_correct": llm_correct,
            "llm_model": verdict.source_model,
            "llm_reasoning": verdict.reasoning,
        })
        mark = "FIXED" if llm_correct else "still wrong"
        logger.info(
            "[%d/%d] (%s) local=%s -> llm=%s [%s]",
            len(rows), len(misses), "escalates" if would_escalate else "would NOT escalate",
            r["predicted"], verdict.label, mark,
        )

    _report(rows)


if __name__ == "__main__":
    main()
