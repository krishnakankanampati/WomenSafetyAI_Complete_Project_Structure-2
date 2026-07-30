"""
MongoDB connection and incident storage for the Women Safety AI pipeline.

Only non-Safe predictions get stored here - Safe comments are ignored per
the pipeline design (see social_media/collector.py).
"""

import os
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "women_safety_ai"

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _client[DB_NAME].processed_comments.create_index("comment_id", unique=True)
    return _client[DB_NAME]


def try_claim(comment_id) -> bool:
    """
    Atomically claims comment_id for processing. Returns True only for the
    caller that wins the race - the unique index on comment_id means at most
    one insert_one across any number of concurrent callers (different
    processes, different asyncio tasks, whatever) can succeed; every other
    caller gets DuplicateKeyError and should treat this comment as already
    handled. Replaces the old is_processed()-then-mark_processed() pair,
    which had a check-then-act gap: two callers could both see "not yet
    processed" before either wrote the mark, and both would classify+alert
    on the same comment (this is exactly what happened in production - two
    concurrent pollers duplicate-alerted on the same comment 4ms apart).
    """
    try:
        get_db().processed_comments.insert_one({"comment_id": comment_id})
        return True
    except DuplicateKeyError:
        return False


def save_oauth_credentials(creds_json: dict) -> None:
    get_db().oauth_credentials.update_one(
        {"_id": "default"},
        {"$set": creds_json},
        upsert=True,
    )


def load_oauth_credentials():
    return get_db().oauth_credentials.find_one({"_id": "default"})


def clear_oauth_credentials() -> None:
    get_db().oauth_credentials.delete_one({"_id": "default"})


def save_meta_credentials(creds_json: dict) -> None:
    get_db().oauth_credentials.update_one(
        {"_id": "meta"},
        {"$set": creds_json},
        upsert=True,
    )


def load_meta_credentials():
    return get_db().oauth_credentials.find_one({"_id": "meta"})


def clear_meta_credentials() -> None:
    get_db().oauth_credentials.delete_one({"_id": "meta"})


def save_incident(platform, video_id, username, comment, comment_time, label, confidence, detected_at, llm_reasoning=None):
    db = get_db()
    doc = {
        "platform": platform,
        "video_id": video_id,
        "username": username,
        "comment": comment,
        "comment_time": comment_time,
        "label": label,
        "confidence": confidence,
        "llm_reasoning": llm_reasoning,
        "detected_at": detected_at,
    }
    result = db.incidents.insert_one(doc)
    return result.inserted_id


def get_incidents(video_id):
    docs = get_db().incidents.find({"video_id": video_id}).sort("detected_at", -1)
    return [
        {
            "id": str(doc["_id"]),
            "platform": doc["platform"],
            "video_id": doc["video_id"],
            "username": doc["username"],
            "comment": doc["comment"],
            "comment_time": doc["comment_time"],
            "label": doc["label"],
            "confidence": doc["confidence"],
            "llm_reasoning": doc.get("llm_reasoning"),
            # pymongo returns a naive datetime (strips tzinfo on read) even
            # though it was saved via datetime.now(timezone.utc) - its wall-
            # clock value IS UTC, so isoformat() without reattaching tzinfo
            # omits the offset entirely, and the frontend's `new Date(...)`
            # then misreads it as local time instead of UTC.
            "detected_at": doc["detected_at"].replace(tzinfo=timezone.utc).isoformat(),
        }
        for doc in docs
    ]
