"""
====================================================
Women Safety AI Project
File : fix_benchmark_labels.py

Corrects mislabeled ground truth in the held-out benchmark
(saved_models/reports/{big_test,fresh50,simple50}_{test_set,results}.csv).

Why this exists: the 76 rows the model got "wrong" were used to score the
LLM fallback (Gemini scored 40/76). Reviewing them against config.CLASS_NAMES
showed a chunk of the *labels* were wrong, not the predictions - so both the
local model and the LLM were being marked wrong for giving the better answer.
Any provider comparison run against the uncorrected benchmark understates
every model tested.

The dominant error is over-application of "Hate Speech" to Indic-language
movie/fan/political commentary that has no group-hate basis at all - the same
failure mode commit 331c525 fixed in the training data, which never reached
these evaluation files.

Corrections are keyed by row index into llm_fallback_verification.csv (the
completed 76-row Gemini run) and resolved to full text at runtime, so the
exact string never has to be duplicated here.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.fix_benchmark_labels --dry-run   # preview
    python -m ai_model.fix_benchmark_labels             # apply
====================================================
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "reports"
VERIFICATION_FILE = REPORTS_DIR / "llm_fallback_verification.csv"
PAIRS = [("big_test_set.csv", "big_test_results.csv"),
         ("fresh50_test_set.csv", "fresh50_results.csv"),
         ("simple50_test_set.csv", "simple50_results.csv")]

# Verification runs record their own copy of `expected`, so they go stale the
# moment the ground truth moves. Kept in sync here rather than left to drift.
#
# IMPORTANT caveat on re-scoring these: the corrections above were made with
# the gemini-3.5-flash-lite verdicts visible, so re-scoring *that* run against
# the corrected labels is circular and flatters it. Treat the resulting number
# as a consistency check, not as a measurement of Gemini's accuracy. Only a
# model whose verdicts were not consulted during the review gives a clean read.
VERIFICATION_FILES = [
    ("llm_fallback_verification.csv", "gemini_verdict", "gemini_correct"),
    ("llm_fallback_verification_flash.csv", "llm_verdict", "llm_correct"),
    ("llm_fallback_verification_hf.csv", "llm_verdict", "llm_correct"),
]

# 1-based row number in llm_fallback_verification.csv -> (corrected label, why)
CORRECTIONS = {
    1:  ("Safe", "'no need to take such people seriously' - dismissive, but names no group and insults no one"),
    2:  ("Safe", "'listening to them is a waste of time' - same: vague dismissal, no group basis"),
    5:  ("Safe", "garbled 'you'll feel bad about the answer he gave' - no group basis anywhere"),
    13: ("Safe", "sarcastic outrage AT the Pollachi perpetrators, not hate toward a group"),
    20: ("Safe", "'dei tejai, what are you doing here' + laughing emojis - friendly banter"),
    30: ("Safe", "casual Pokkiri Raja movie reference"),
    35: ("Offensive", "'you are a pipe' IS name-calling - was labelled Safe, too lenient"),
    36: ("Safe", "Ambedkar / Dalit Praja Sena political discussion, no slur or attack"),
    47: ("Safe", "'Rasmika' + wilted-rose emoji - a name and an emoji"),
    50: ("Safe", "'give me a list of Chinese apps so I don't install them' - consumer question"),
    60: ("Safe", "'SHYATA YASH' - just names"),
    68: ("Safe", "Pawan Kalyan fan praise, no target at all"),
    32: ("Threat", "'if anyone lays hands on women, cut off hands/head' - violent, but PRO-women; not group hate"),
    33: ("Offensive", "insults people who start caste fights - an accusation, not a protected group"),
    34: ("Sexual Harassment", "objectifies a woman's body and references an actress sexually"),
    44: ("Offensive", "vulgar abuse aimed at one named individual - no group dimension"),
    59: ("Offensive", "rambling insult at an individual, not a group"),
}

# Reviewed and deliberately NOT changed - recorded so a future pass doesn't
# redo the same analysis. Each is a genuine judgement call where the existing
# label is at least as defensible as the alternative.
LEFT_ALONE = {
    3:  "'no one can do anything to fools' - generic aphorism, Safe/Offensive both arguable",
    4:  "'twitter tillu gaabii baanisa' - too garbled to label confidently",
    7:  "Madurai/slipper movie-caste discourse - Offensive vs Hate Speech genuinely unclear",
    17: "'what Tamil is this... makes me laugh' - mild mockery, right on the Safe/Offensive line",
    19: "vulgar virginity question - Offensive label vs Gemini's Sexual Harassment, both defensible",
    23: "body-shaming with a sexual reference - Sexual Harassment vs Offensive, borderline",
    25: "'giving slippers to the dogs' - figurative violence, Threat vs Offensive",
    43: "'China chakkas' - 'chakka' IS a slur; keeps it consistent with rows 45/46",
    49: "'you should be beaten' - Offensive vs Threat, borderline",
    51: "cursing a group's birth - Offensive vs Hate Speech, borderline",
    54: "genital reference at a named woman - Sexual Harassment label is right, both models under-called it",
    55: "'should be beaten and killed' - Threat label is right, both models under-called it",
    58: "'don't comment in other languages, you tigers' - language-based but not demeaning",
    62: "unclear insult at an individual - Offensive vs Hate Speech, weak either way",
}


def main():
    ap = argparse.ArgumentParser(description="Correct mislabeled benchmark ground truth")
    ap.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = ap.parse_args()

    ver = pd.read_csv(VERIFICATION_FILE, encoding="utf-8-sig")
    fixes = {}
    for idx, (new_label, reason) in CORRECTIONS.items():
        row = ver.iloc[idx - 1]
        fixes[str(row["text"])] = (str(row["expected"]), new_label, reason, str(row["language"]))

    logger.info("=" * 70)
    logger.info("%d corrections prepared (%d rows reviewed and left alone)",
                len(fixes), len(LEFT_ALONE))
    logger.info("=" * 70)
    for text, (old, new, reason, lang) in fixes.items():
        logger.info("[%-13s] %-18s -> %-18s", lang, old, new)
        logger.info("    %s", text[:88])
        logger.info("    why: %s", reason)

    total_changed = 0
    accuracy_lines = []
    for set_name, results_name in PAIRS:
        for name in (set_name, results_name):
            path = REPORTS_DIR / name
            if not path.exists():
                logger.warning("Missing, skipped: %s", name)
                continue
            df = pd.read_csv(path, encoding="utf-8-sig")
            before = df["correct"].mean() if "correct" in df.columns else None

            mask = df["text"].astype(str).isin(fixes)
            if mask.any():
                df.loc[mask, "expected"] = df.loc[mask, "text"].astype(str).map(
                    lambda t: fixes[t][1]
                )
                # results files carry a derived `correct` column - recompute it,
                # or every downstream score silently keeps the stale value.
                if "correct" in df.columns and "predicted" in df.columns:
                    df["correct"] = df["predicted"].astype(str) == df["expected"].astype(str)
                total_changed += int(mask.sum())
                logger.info("%-26s %d rows relabelled%s", name, int(mask.sum()),
                            " (dry run)" if args.dry_run else "")
                if not args.dry_run:
                    df.to_csv(path, index=False, encoding="utf-8-sig")

            # Reported from the in-memory frame, not by re-reading: under
            # --dry-run nothing was written, so a re-read would report the old
            # numbers as if they were the new ones.
            if before is not None:
                accuracy_lines.append(
                    "%-26s %d/%d (%.1f%%)  was %.1f%%"
                    % (name, df["correct"].sum(), len(df),
                       100 * df["correct"].mean(), 100 * before)
                )

    for name, verdict_col, correct_col in VERIFICATION_FILES:
        path = REPORTS_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        mask = df["text"].astype(str).isin(fixes)
        if not mask.any():
            continue
        before = df[correct_col].sum()
        df.loc[mask, "expected"] = df.loc[mask, "text"].astype(str).map(lambda t: fixes[t][1])
        df[correct_col] = df[verdict_col].astype(str) == df["expected"].astype(str)
        total_changed += int(mask.sum())
        logger.info("%-36s %d rows relabelled; %s %d -> %d%s", name, int(mask.sum()),
                    correct_col, before, df[correct_col].sum(),
                    " (dry run)" if args.dry_run else "")
        if not args.dry_run:
            df.to_csv(path, index=False, encoding="utf-8-sig")

    logger.info("-" * 70)
    logger.info("Total row updates across all files: %d%s", total_changed,
                " (dry run - nothing written)" if args.dry_run else "")
    logger.info("Local model accuracy against the corrected ground truth:")
    for line in accuracy_lines:
        logger.info("  %s", line)
    logger.info("NOTE: LLM scores above are circular - see VERIFICATION_FILES comment.")


if __name__ == "__main__":
    main()
