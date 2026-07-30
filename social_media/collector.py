"""
Milestone 5: given a YouTube video URL, continuously poll for new comments,
classify each one with the trained safety model (+ Gemini fallback for low
confidence), save non-Safe comments to MongoDB as incidents, and email the
NGO contact. Already-processed comments are skipped on later polls.

Usage:
    One-shot (existing behavior):
        python -m social_media.collector --url "https://www.youtube.com/watch?v=abc123"

    Continuous polling every 5 seconds:
        python -m social_media.collector --url "https://www.youtube.com/watch?v=abc123" --watch --interval 5
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ai_model.llm_fallback import classify_with_llm
from ai_model.predict import get_predictor
from database.mongodb import save_incident, try_claim
from email_service.send_email import send_alert_email
from social_media.config import YOUTUBE_API_KEY

PLATFORM = "YouTube"


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be",):
        video_id = parsed.path.lstrip("/")
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/")[2]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = None
    else:
        video_id = None

    if not video_id:
        raise ValueError(f"Could not extract a video ID from URL: {url}")
    return video_id


def fetch_comments(video_id: str, max_results: int = 50):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results, 100),
        textFormat="plainText",
        order="time",
    )
    while request is not None and len(comments) < max_results:
        response = request.execute()
        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]
            snippet = top_comment["snippet"]
            comments.append({
                "comment_id": top_comment["id"],
                "username": snippet["authorDisplayName"],
                "comment": snippet["textDisplay"],
                "comment_time": snippet["publishedAt"],
            })
        request = youtube.commentThreads().list_next(request, response)

    return comments[:max_results]


def process_video(video_id: str, max_results: int = 50) -> int:
    """Fetch, classify, and act on comments not seen before. Returns incident count."""
    comments = fetch_comments(video_id, max_results=max_results)
    # try_claim is atomic (unique index in Mongo) - the filter itself is what
    # prevents two concurrent callers (e.g. a CLI watch loop left running
    # alongside the dashboard) from both classifying+alerting on the same
    # comment, not anything downstream in this function.
    new_comments = [c for c in comments if try_claim(c["comment_id"])]
    if not new_comments:
        return 0

    predictor = get_predictor()
    predictions = predictor.predict_batch([c["comment"] for c in new_comments])

    incidents = 0
    for c, pred in zip(new_comments, predictions):
        marker = " [LLM]" if pred.source == "llm_escalated" else ""
        print(f"[{pred.label:<18}]{marker} conf={pred.confidence:.2f} | {c['username']}: {c['comment']}")

        if pred.label != "Safe":
            reasoning = pred.llm_reasoning
            if reasoning is None:
                # High local confidence skipped escalation, so there's no
                # explanation to show yet - the label stays the local
                # model's (already confident), this call is only to get
                # human-readable reasoning text for the incident/email.
                verdict = classify_with_llm(c["comment"])
                if verdict is not None:
                    reasoning = verdict.reasoning

            # Generated once and passed to both calls so the incident record
            # and the alert email always agree on exactly when this was
            # detected, rather than each taking its own now() a moment apart.
            detected_at = datetime.now(timezone.utc)

            save_incident(
                platform=PLATFORM,
                video_id=video_id,
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
                video_id=video_id,
                username=c["username"],
                comment=c["comment"],
                comment_time=c["comment_time"],
                label=pred.label,
                confidence=pred.confidence,
                detected_at=detected_at,
                llm_reasoning=reasoning,
            )
            incidents += 1

    return incidents


def watch(video_id: str, interval: int, max_results: int = 50):
    print(f"Watching video {video_id} every {interval}s (Ctrl+C to stop)...\n")
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            try:
                incidents = process_video(video_id, max_results=max_results)
                if incidents:
                    print(f"[{timestamp}] {incidents} new incident(s) alerted.")
            except HttpError as e:
                reason = e.error_details[0]["reason"] if e.error_details else str(e)
                print(f"[{timestamp}] YouTube API error ({reason}), will retry next cycle.", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Fetch and classify comments for a YouTube video")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--max", type=int, default=50, help="Max comments to fetch per poll")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new comments instead of running once")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between polls in --watch mode")
    args = parser.parse_args()

    if not YOUTUBE_API_KEY:
        print(
            "YOUTUBE_API_KEY not set. Add it to .env (see .env.example) and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    video_id = extract_video_id(args.url)
    print(f"Video ID: {video_id}")

    if args.watch:
        watch(video_id, interval=args.interval, max_results=args.max)
        return

    try:
        incidents = process_video(video_id, max_results=args.max)
    except HttpError as e:
        reason = e.error_details[0]["reason"] if e.error_details else str(e)
        print(f"YouTube API error ({reason}): comments may be disabled for this video.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{incidents} incident(s) saved to MongoDB and emailed to the NGO contact.")


if __name__ == "__main__":
    main()
