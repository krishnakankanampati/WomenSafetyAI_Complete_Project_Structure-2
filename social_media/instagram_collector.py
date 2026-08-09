"""
Instagram counterpart to social_media/collector.py's process_video: given an
Instagram media id, continuously poll for new comments, classify, save
incidents, and email the NGO contact. Reuses every pipeline piece unmodified
- only the fetch step and PLATFORM differ.

Usage:
    python -m social_media.instagram_collector --media-id 17912345678901234
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import requests

from ai_model.llm_fallback import classify_with_llm
from ai_model.predict import get_predictor
from backend.meta_oauth_store import GRAPH_BASE, get_valid_meta_credentials
from database.mongodb import save_incident, try_claim
from email_service.send_email import send_alert_email

PLATFORM = "Instagram"


def _ig_page_access_token(pages: list[dict]) -> str:
    """v1 assumption (see plan): exactly one connected Page has an Instagram
    Business Account linked, so media ids don't need to carry a page
    reference the way Facebook post ids do - just use that Page's token."""
    for page in pages:
        if page.get("instagram_business_account_id"):
            return page["access_token"]
    raise ValueError("No connected Page has a linked Instagram Business Account")


def fetch_comments(media_id: str, max_results: int = 50):
    creds = get_valid_meta_credentials()
    if creds is None:
        raise RuntimeError("Meta (Facebook/Instagram) not connected - visit /auth/meta/login first")

    access_token = _ig_page_access_token(creds["pages"])
    comments = []

    url = f"{GRAPH_BASE}/{media_id}/comments"
    params = {
        "fields": "id,text,username,timestamp",
        "access_token": access_token,
        "limit": min(max_results, 100),
    }
    while url and len(comments) < max_results:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("data", []):
            comments.append({
                "comment_id": item["id"],
                "username": item.get("username", "Unknown"),
                "comment": item.get("text", ""),
                "comment_time": item["timestamp"],
            })
        url = data.get("paging", {}).get("next")
        params = None  # the "next" url already carries every param

    return comments[:max_results]


def process_media(media_id: str, max_results: int = 50) -> tuple[int, int]:
    """Fetch, classify, and act on comments not seen before. Returns (incident_count, total_comments_fetched)."""
    comments = fetch_comments(media_id, max_results=max_results)
    new_comments = [c for c in comments if try_claim(c["comment_id"])]
    if not new_comments:
        return 0, len(comments)

    predictor = get_predictor()
    predictions = predictor.predict_batch([c["comment"] for c in new_comments])

    incidents = 0
    for c, pred in zip(new_comments, predictions):
        marker = " [LLM]" if pred.source == "llm_escalated" else ""
        print(f"[{pred.label:<18}]{marker} conf={pred.confidence:.2f} | {c['username']}: {c['comment']}")

        if pred.label != "Safe":
            reasoning = pred.llm_reasoning
            if reasoning is None:
                verdict = classify_with_llm(c["comment"])
                if verdict is not None:
                    reasoning = verdict.reasoning

            detected_at = datetime.now(timezone.utc)

            save_incident(
                platform=PLATFORM,
                video_id=media_id,
                username=c["username"],
                comment=c["comment"],
                comment_time=c["comment_time"],
                label=pred.label,
                confidence=pred.confidence,
                detected_at=detected_at,
                llm_reasoning=reasoning,
            )
            send_alert_email(
                platform=PLATFORM,
                video_id=media_id,
                username=c["username"],
                comment=c["comment"],
                comment_time=c["comment_time"],
                label=pred.label,
                confidence=pred.confidence,
                detected_at=detected_at,
                llm_reasoning=reasoning,
            )
            incidents += 1

    return incidents, len(comments)


def watch(media_id: str, interval: int, max_results: int = 50):
    print(f"Watching Instagram media {media_id} every {interval}s (Ctrl+C to stop)...\n")
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            try:
                incidents, _ = process_media(media_id, max_results=max_results)
                if incidents:
                    print(f"[{timestamp}] {incidents} new incident(s) alerted.")
            except requests.HTTPError as e:
                print(f"[{timestamp}] Instagram API error ({e}), will retry next cycle.", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Fetch and classify comments for an Instagram media item")
    parser.add_argument("--media-id", required=True, help="Instagram media id")
    parser.add_argument("--max", type=int, default=50, help="Max comments to fetch per poll")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new comments instead of running once")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between polls in --watch mode")
    args = parser.parse_args()

    if args.watch:
        watch(args.media_id, interval=args.interval, max_results=args.max)
        return

    incidents, total = process_media(args.media_id, max_results=args.max)
    print(f"\n{incidents} incident(s) saved to MongoDB and emailed to the NGO contact ({total} comment(s) checked).")


if __name__ == "__main__":
    main()
