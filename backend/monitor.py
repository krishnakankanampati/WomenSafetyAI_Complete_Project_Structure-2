"""
Background polling registry: one asyncio task per (platform, content_id)
pair, dispatching to the already-proven per-platform process_* functions
unmodified. One task per key (guarded by start_monitor's no-op-if-running
check) is what actually prevents the check-then-act dedup race in each
collector's process_* from double-firing an incident/email - not anything in
the collectors themselves.
"""

import asyncio
import logging

from social_media.collector import process_video
from social_media.facebook_collector import process_post
from social_media.instagram_collector import process_media

logger = logging.getLogger(__name__)

PROCESSORS = {
    "youtube": process_video,
    "instagram": process_media,
    "facebook": process_post,
}

# Instagram/Facebook polling shares one ~200 calls/user/hour Meta rate-limit
# budget across every connected item - a 5s interval (fine for YouTube, which
# has its own separate quota) would blow through it fast with more than one
# item polled.
DEFAULT_INTERVALS = {
    "youtube": 5,
    "instagram": 30,
    "facebook": 30,
}

_monitor_tasks: dict[str, asyncio.Task] = {}

# Latest total-comments-on-the-platform reading per (platform, content_id),
# refreshed every poll. Deliberately not persisted to Mongo: it's a live
# "what does the platform report right now" number, not a historical count -
# process_*'s own dedup (try_claim) already prevents re-alerting on comments
# we've seen, but every poll still re-fetches and re-counts all of them.
_last_total_comments: dict[str, int] = {}


def _key(platform: str, content_id: str) -> str:
    return f"{platform}:{content_id}"


async def _poll_loop(platform: str, content_id: str, max_results: int = 50, interval: int | None = None):
    process = PROCESSORS[platform]
    interval = interval if interval is not None else DEFAULT_INTERVALS[platform]
    key = _key(platform, content_id)
    while True:
        try:
            _, total = await asyncio.to_thread(process, content_id, max_results)
            _last_total_comments[key] = total
        except Exception:
            logger.exception("poll failed for %s:%s", platform, content_id)
        await asyncio.sleep(interval)


def start_monitor(platform: str, content_id: str) -> bool:
    """Returns False (no-op) if a monitor for this (platform, content_id) is already running."""
    key = _key(platform, content_id)
    existing = _monitor_tasks.get(key)
    if existing and not existing.done():
        return False
    _monitor_tasks[key] = asyncio.create_task(_poll_loop(platform, content_id))
    return True


def stop_monitor(platform: str, content_id: str) -> bool:
    """Returns False if no monitor was running for this (platform, content_id)."""
    task = _monitor_tasks.pop(_key(platform, content_id), None)
    _last_total_comments.pop(_key(platform, content_id), None)
    if task is None:
        return False
    task.cancel()
    return True


def is_monitoring(platform: str, content_id: str) -> bool:
    task = _monitor_tasks.get(_key(platform, content_id))
    return task is not None and not task.done()


def get_total_comments(platform: str, content_id: str) -> int | None:
    """Last polled total comment count, or None if no poll has completed yet."""
    return _last_total_comments.get(_key(platform, content_id))
