"""
====================================================
Women Safety AI Project
File : llm_fallback.py

Second opinion for predict.py when the local classifier's confidence
falls below config.LLM_ESCALATION_THRESHOLD. Calls Gemini with the same
5-category taxonomy the local model uses, constrained to a valid label
via a JSON schema, so the response never needs free-text parsing.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. If it's
missing, or the call fails for any reason (network, rate limit, invalid
key), this returns None and the caller keeps the local model's
prediction - the fallback is a confidence booster for uncertain cases,
never a hard dependency.
====================================================
"""

import logging
import os
import time
from typing import Literal, Optional

from google import genai
from google.genai import errors
from pydantic import BaseModel

from ai_model.config import LLM_FALLBACK_MODELS, QUOTA_RETRY_AFTER_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a content moderation classifier for a women's safety app used across India. Comments arrive in English, Telugu, Tamil, Kannada - in native script, romanized/transliterated (e.g. "nuvvu ekkada unnav"), or code-mixed with English (Tanglish, Kanglish, Roman Telugu).

Classify each comment into exactly one of these 5 categories:

- Safe: benign content - greetings, compliments without ulterior motive, general discussion, unrelated chat.
- Offensive: insults, name-calling, vulgar language directed at a person, without a sexual or group-hate dimension.
- Sexual Harassment: unwanted sexual attention, requests for photos/meetups with sexual undertone, objectifying comments, stalking-adjacent requests (e.g. "send your address", "meet me alone"). Watch for compliment-then-request patterns ("you look beautiful, send me your photo") - the compliment alone is Safe, but paired with a request or clear romantic/sexual escalation it becomes Sexual Harassment.
- Threat: explicit or implied intent to harm, kill, stalk, or retaliate against a specific person ("I know where you live", "you'll regret this").
- Hate Speech: content demeaning a group based on caste, religion, gender, ethnicity, or community (not an individual) - e.g. "that caste is inferior", "women don't deserve rights", "all X people are Y". Distinguish from Offensive (targets an individual, no group basis) and from ordinary political/critical opinion (no group basis).

Some comments are genuinely ambiguous (e.g. caste-politics movie discourse, sarcasm, coded language). Make your best judgment and explain briefly why."""


class ModerationVerdict(BaseModel):
    label: Literal[
        "Safe", "Offensive", "Sexual Harassment", "Threat", "Hate Speech"
    ]
    reasoning: str
    # Which model in the chain actually answered. Filled in after validation,
    # never by the LLM - see RESPONSE_SCHEMA, which strips it before the schema
    # is sent to the API so the model is not asked to invent a value.
    source_model: str = ""


def _response_schema():
    schema = ModerationVerdict.model_json_schema()
    schema.get("properties", {}).pop("source_model", None)
    schema["required"] = [f for f in schema.get("required", []) if f != "source_model"]
    return schema


RESPONSE_SCHEMA = _response_schema()


# Distinguishes "every model's quota is spent, stop trying" from a hard auth
# failure. verify_llm_fallback.py checks this to decide whether waiting and
# retrying is worth it - a 429 clears at the next daily reset, a 401 never
# clears on its own.
class QuotaExhausted(Exception):
    pass


_client = None
_disabled = False  # set after a hard auth failure so we don't retry every request
_benched = {}      # model name -> time.monotonic() when it last returned 429
_dead = set()      # models that 404'd: renamed, retired, or a typo in the chain


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _available_models():
    """
    Models worth trying right now, preferred first.

    Two kinds of exclusion, deliberately different:
      - benched (429): temporary. Skipped until QUOTA_RETRY_AFTER_SECONDS has
        passed - long enough for a per-minute limit to clear, short enough that
        a daily quota resetting mid-process gets picked back up.
      - dead (404): permanent for this process. A model that doesn't exist -
        retired, renamed, or a typo in the chain - will never start working, so
        retrying it just adds a wasted round-trip to every single request.
    """
    now = time.monotonic()
    return [
        m for m in LLM_FALLBACK_MODELS
        if m not in _dead
        and now - _benched.get(m, float("-inf")) >= QUOTA_RETRY_AFTER_SECONDS
    ]


def _call(model: str, text: str) -> ModerationVerdict:
    """One attempt against one model. Raises; callers decide what to swallow."""
    response = _get_client().models.generate_content(
        model=model,
        contents=text,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_json_schema": RESPONSE_SCHEMA,
        },
    )
    verdict = ModerationVerdict.model_validate_json(response.text)
    verdict.source_model = model
    return verdict


def classify_with_llm(text: str) -> Optional[ModerationVerdict]:
    """
    Ask Gemini to classify `text` into the same taxonomy as the local
    model. Returns None (never raises) if the fallback is unavailable or
    the call fails - callers should treat None as "keep the local
    prediction unchanged".
    """
    global _disabled

    if _disabled:
        return None

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        logger.warning(
            "LLM fallback disabled for this run: no GEMINI_API_KEY (or "
            "GOOGLE_API_KEY) set. Set one to enable low-confidence escalation."
        )
        _disabled = True
        return None

    candidates = _available_models()
    if not candidates:
        logger.warning(
            "LLM fallback skipped: every model in the chain is quota-benched "
            "(retrying each after %ds).", QUOTA_RETRY_AFTER_SECONDS,
        )
        return None

    for i, model in enumerate(candidates):
        try:
            verdict = _call(model, text)
            if i > 0:
                logger.info("Answered by %s after %d model(s) failed.", model, i)
            return verdict

        except errors.ClientError as e:
            code = getattr(e, "code", None)
            if code in (401, 403):
                # An auth failure is about the key, not the model - trying the
                # rest of the chain would fail identically.
                logger.warning(
                    "LLM fallback disabled for this run: GEMINI_API_KEY was rejected "
                    "(%s). Check the key's validity to enable low-confidence escalation.",
                    e,
                )
                _disabled = True
                return None
            if code == 429:
                _benched[model] = time.monotonic()
                logger.warning("%s hit quota - trying next model in the chain.", model)
                continue
            if code == 404:
                _dead.add(model)
                logger.warning(
                    "%s does not exist on this API version - dropped from the chain "
                    "for this run. Fix LLM_FALLBACK_MODELS in config.py.", model,
                )
                continue
            logger.warning("%s skipped (client error %s): %s", model, code, str(e)[:120])
            continue

        except errors.ServerError as e:
            # 503 "high demand" is capacity, not quota - another model on
            # different hardware may well answer, so keep going.
            logger.warning("%s unavailable (server error): %s", model, str(e)[:120])
            continue

        except errors.APIError as e:
            logger.warning("%s skipped (API error): %s", model, str(e)[:120])
            continue

        except Exception as e:
            # Catch-all so an unconfigured key (raised at client construction -
            # exact exception type isn't guaranteed across SDK versions) or any
            # other unexpected failure never crashes the caller. The contract
            # here is "never raise" - a fallback that can itself take down
            # inference defeats the point of being a fallback.
            logger.warning("%s skipped (unexpected error): %s", model, str(e)[:120])
            continue

    logger.warning("LLM fallback skipped: all %d model(s) failed.", len(candidates))
    return None


def classify_or_raise(text: str) -> Optional[ModerationVerdict]:
    """
    Same as classify_with_llm, but raises QuotaExhausted once *every* model in
    the chain has hit its quota, instead of folding that into None. Only for
    batch/offline callers (verification sweeps) that want to wait out a
    per-minute limit and resume; production inference should stay on
    classify_with_llm, which never raises.

    Non-quota errors from the last model propagate, so a sweep still surfaces
    genuine breakage rather than silently treating it as exhausted quota.
    """
    if _disabled:
        return None

    candidates = _available_models()
    if not candidates:
        raise QuotaExhausted(
            "all %d models quota-benched" % len(LLM_FALLBACK_MODELS)
        )

    last_error = None
    for i, model in enumerate(candidates):
        try:
            verdict = _call(model, text)
            if i > 0:
                logger.info("Answered by %s after %d model(s) failed.", model, i)
            return verdict
        except errors.ClientError as e:
            code = getattr(e, "code", None)
            if code == 429:
                _benched[model] = time.monotonic()
                logger.warning("%s hit quota - trying next model in the chain.", model)
                last_error = QuotaExhausted(str(e))
                continue
            if code == 404:
                _dead.add(model)
                logger.warning("%s does not exist - dropped from the chain.", model)
                last_error = e
                continue
            raise
        except errors.ServerError as e:
            logger.warning("%s unavailable (server error) - trying next model.", model)
            last_error = e
            continue

    raise last_error
