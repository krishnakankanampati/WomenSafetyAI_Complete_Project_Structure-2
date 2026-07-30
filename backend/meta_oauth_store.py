"""
Meta (Instagram + Facebook) OAuth glue. One login covers both platforms -
they share the same Meta Developer App, the same Facebook Login dialog, and
the same GET /me/accounts call, so there's one shared credential doc rather
than two. Unlike Google's flow (backend/oauth_store.py), Meta's OAuth is
plain server-side code exchange - no PKCE, no comparable SDK - so this talks
to the Graph API directly via requests.
"""

from datetime import datetime, timedelta, timezone

import requests

from backend.config import (
    META_APP_ID,
    META_APP_SECRET,
    META_OAUTH_REDIRECT_URI,
    META_OAUTH_SCOPES,
    META_PAGE_ACCESS_TOKEN,
)
from database.mongodb import (
    clear_meta_credentials,
    load_meta_credentials,
    save_meta_credentials,
)

GRAPH_API_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Refresh this many days before the long-lived token's actual expiry - there's
# no refresh_token concept like Google's, only re-exchanging a still-valid
# token for a fresh ~60-day one, and nothing here runs on an independent
# schedule, so refresh generously early rather than cutting it close.
REFRESH_THRESHOLD_DAYS = 14


def build_authorization_url() -> str:
    return (
        f"https://www.facebook.com/{GRAPH_API_VERSION}/dialog/oauth"
        f"?client_id={META_APP_ID}"
        f"&redirect_uri={META_OAUTH_REDIRECT_URI}"
        f"&scope={META_OAUTH_SCOPES}"
        f"&response_type=code"
    )


def exchange_code_for_token(code: str) -> dict:
    resp = requests.get(
        f"{GRAPH_BASE}/oauth/access_token",
        params={
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "redirect_uri": META_OAUTH_REDIRECT_URI,
            "code": code,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def exchange_for_long_lived_token(short_lived_token: str) -> dict:
    """Also the refresh mechanism: re-calling this with a still-valid
    long-lived token yields a fresh ~60-day one."""
    resp = requests.get(
        f"{GRAPH_BASE}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "fb_exchange_token": short_lived_token,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_pages(user_access_token: str) -> list[dict]:
    """Every Page the user administers, each with its own Page Access Token
    and (if linked) an Instagram Business Account id. Page-scoped tokens are
    used for calls against /{page-id}/... and IG business-account edges per
    Meta's guidance, not the raw user token."""
    resp = requests.get(
        f"{GRAPH_BASE}/me/accounts",
        params={
            "fields": "id,name,access_token,instagram_business_account",
            "access_token": user_access_token,
        },
        timeout=10,
    )
    resp.raise_for_status()
    pages = []
    for item in resp.json().get("data", []):
        ig_account = item.get("instagram_business_account")
        pages.append({
            "id": item["id"],
            "name": item["name"],
            "access_token": item["access_token"],
            "instagram_business_account_id": ig_account["id"] if ig_account else None,
        })
    return pages


def _save(user_access_token: str, expires_in_seconds: int, pages: list[dict]) -> None:
    save_meta_credentials({
        "user_access_token": user_access_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds),
        "pages": pages,
    })


def connect(code: str) -> None:
    """Full flow: authorization code -> long-lived token -> Pages -> save."""
    short_lived = exchange_code_for_token(code)
    long_lived = exchange_for_long_lived_token(short_lived["access_token"])
    pages = fetch_pages(long_lived["access_token"])
    _save(long_lived["access_token"], long_lived.get("expires_in", 60 * 24 * 3600), pages)


def _credentials_from_env_token() -> dict | None:
    """Builds the same credential shape from a Page Access Token pasted
    straight into .env as META_PAGE_ACCESS_TOKEN, skipping OAuth entirely.

    This is the supported path for Meta, not a workaround: "Facebook Login
    for Business" locks "Enforce HTTPS" on for redirect URIs (the toggle is
    greyed out in the app console), so it rejects http://localhost:8000/...
    outright - the browser flow simply cannot complete against a local dev
    server without an HTTPS tunnel whose URL changes on every restart. And
    since this dashboard only ever monitors its own owner's Page and linked
    Instagram account, a per-user consent flow buys nothing anyway.

    Page Access Tokens derived from a long-lived user token don't expire, so
    there's no refresh to schedule - see this module's __main__ block for the
    one-off exchange that mints one.
    """
    if not META_PAGE_ACCESS_TOKEN:
        return None

    # A Page token makes /me resolve to the Page itself, not the user, which
    # is exactly the id/name/IG-link trio the pages list needs.
    resp = requests.get(
        f"{GRAPH_BASE}/me",
        params={
            "fields": "id,name,instagram_business_account",
            "access_token": META_PAGE_ACCESS_TOKEN,
        },
        timeout=10,
    )
    resp.raise_for_status()
    page = resp.json()
    ig_account = page.get("instagram_business_account")

    return {
        "user_access_token": META_PAGE_ACCESS_TOKEN,
        "pages": [{
            "id": page["id"],
            "name": page["name"],
            "access_token": META_PAGE_ACCESS_TOKEN,
            "instagram_business_account_id": ig_account["id"] if ig_account else None,
        }],
    }


def get_valid_meta_credentials() -> dict | None:
    """Returns the stored credential doc, refreshing the token if it's within
    REFRESH_THRESHOLD_DAYS of expiring. Falls back to the .env Page token
    when no OAuth doc exists. Returns None if neither is configured, or if a
    stored OAuth token has already lapsed (no silent recovery path - full
    re-login via /auth/meta/login would be required in that case)."""
    doc = load_meta_credentials()
    if doc is None:
        return _credentials_from_env_token()

    expires_at = doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) >= expires_at:
        return None

    if expires_at - datetime.now(timezone.utc) < timedelta(days=REFRESH_THRESHOLD_DAYS):
        refreshed = exchange_for_long_lived_token(doc["user_access_token"])
        pages = fetch_pages(refreshed["access_token"])
        _save(refreshed["access_token"], refreshed.get("expires_in", 60 * 24 * 3600), pages)
        doc = load_meta_credentials()

    return doc


def disconnect() -> None:
    clear_meta_credentials()


def _print_page_tokens(user_access_token: str) -> None:
    pages = fetch_pages(user_access_token)

    if not pages:
        print("No Pages found for this token.\n"
              "A personal user token only sees Pages you hold directly - once a Page is\n"
              "owned by a Business Portfolio it disappears from /me/accounts. Use a\n"
              "system-user token instead (Business Settings -> Users -> System users,\n"
              "assign the Page and Instagram account, then Generate new token) and pass\n"
              "it with --system-token.")
        return

    for page in pages:
        ig = page["instagram_business_account_id"]
        print(f"\nPage: {page['name']} (id {page['id']})")
        print(f"Instagram Business Account: {ig or 'not linked'}")
        print("META_PAGE_ACCESS_TOKEN=" + page["access_token"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the non-expiring Page Access Token to put in .env",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--exchange", metavar="USER_TOKEN",
                        help="short-lived user token (Graph API Explorer -> 'Get User "
                             "Access Token'); gets upgraded to long-lived first. Only "
                             "works while the Page is held personally, not by a "
                             "Business Portfolio.")
    source.add_argument("--system-token", metavar="SYSTEM_USER_TOKEN",
                        help="system-user token from Business Settings -> Users -> "
                             "System users. Already permanent, so it's used as-is - and "
                             "it can see Business-Portfolio-owned Pages, which a "
                             "personal user token cannot.")
    args = parser.parse_args()

    if args.exchange:
        _print_page_tokens(exchange_for_long_lived_token(args.exchange)["access_token"])
    else:
        _print_page_tokens(args.system_token)
