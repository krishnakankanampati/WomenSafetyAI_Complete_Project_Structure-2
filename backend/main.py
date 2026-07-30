# FastAPI entry point

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from googleapiclient.discovery import build as build_youtube_client

from ai_model.predict import get_predictor
from backend.config import FRONTEND_URL
from backend.meta_oauth_store import (
    GRAPH_BASE,
    build_authorization_url,
    connect as meta_connect,
    disconnect as meta_disconnect,
    get_valid_meta_credentials,
)
from backend.monitor import PROCESSORS, is_monitoring, start_monitor, stop_monitor
from backend.oauth_store import (
    build_flow,
    get_valid_credentials,
    pop_pending_verifier,
    save_credentials,
    set_pending_verifier,
)
from database.mongodb import clear_oauth_credentials, get_incidents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    # Vite silently moves to 5174, 5175, ... when its default port is taken
    # (a stale dev server, a second terminal), and a hardcoded FRONTEND_URL
    # then blocks every request from the page the user is actually looking at
    # - which surfaces as the UI claiming nothing is configured, since the
    # fetches fail rather than returning false. Any localhost port is fine
    # here: this dashboard is local-only and holds no cookie to steal
    # (allow_credentials stays False).
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.on_event("startup")
def _load_model_at_startup():
    get_predictor()


@app.get("/auth/login")
def auth_login():
    flow = build_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )
    set_pending_verifier(flow.code_verifier)
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
def auth_callback(code: str):
    flow = build_flow(code_verifier=pop_pending_verifier())
    flow.fetch_token(code=code)
    save_credentials(flow.credentials)
    # ?login=success marks this specific redirect (not a plain reload/fresh
    # visit) so the frontend can tell "just finished the login flow" apart
    # from "opening the URL fresh" - see App.jsx's sessionStorage handling.
    return RedirectResponse(f"{FRONTEND_URL}?login=success")


@app.get("/auth/status")
def auth_status():
    creds = get_valid_credentials()
    return {"logged_in": creds is not None}


@app.post("/auth/logout")
def auth_logout():
    clear_oauth_credentials()
    return {"logged_in": False}


@app.get("/auth/meta/login")
def auth_meta_login():
    return RedirectResponse(build_authorization_url())


@app.get("/auth/meta/callback")
def auth_meta_callback(code: str):
    meta_connect(code)
    # Instagram and Facebook share this one redirect - see App.jsx's
    # sessionStorage handling for the shared "meta" login marker.
    return RedirectResponse(f"{FRONTEND_URL}?login=success&platform=meta")


@app.get("/auth/instagram/status")
def auth_instagram_status():
    creds = get_valid_meta_credentials()
    logged_in = creds is not None and any(p["instagram_business_account_id"] for p in creds["pages"])
    return {"logged_in": logged_in}


@app.get("/auth/facebook/status")
def auth_facebook_status():
    creds = get_valid_meta_credentials()
    logged_in = creds is not None and len(creds["pages"]) > 0
    return {"logged_in": logged_in}


@app.post("/auth/instagram/logout")
@app.post("/auth/facebook/logout")
def auth_meta_logout():
    # Aliases of each other by design - disconnecting Meta disconnects both
    # platforms at once, there's no way to disconnect only one. Surfaced
    # plainly in the frontend's PlatformHub copy, not hidden.
    meta_disconnect()
    return {"logged_in": False}


@app.get("/api/videos")
def list_videos():
    creds = get_valid_credentials()
    if creds is None:
        raise HTTPException(status_code=401, detail="Not logged in")

    youtube = build_youtube_client("youtube", "v3", credentials=creds)

    channels_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
    uploads_playlist_id = channels_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    items_resp = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_playlist_id,
        maxResults=25,
    ).execute()

    videos = [
        {
            "video_id": item["snippet"]["resourceId"]["videoId"],
            "title": item["snippet"]["title"],
            "thumbnail_url": item["snippet"]["thumbnails"].get("default", {}).get("url"),
            "published_at": item["snippet"]["publishedAt"],
        }
        for item in items_resp.get("items", [])
    ]
    return {"videos": videos}


@app.get("/api/instagram/media")
def list_instagram_media():
    creds = get_valid_meta_credentials()
    if creds is None:
        raise HTTPException(status_code=401, detail="Not connected")
    ig_page = next((p for p in creds["pages"] if p["instagram_business_account_id"]), None)
    if ig_page is None:
        raise HTTPException(status_code=401, detail="No Instagram Business Account linked")

    resp = requests.get(
        f"{GRAPH_BASE}/{ig_page['instagram_business_account_id']}/media",
        params={
            "fields": "id,caption,media_url,thumbnail_url,timestamp,permalink",
            "access_token": ig_page["access_token"],
        },
        timeout=10,
    )
    resp.raise_for_status()
    return {"media": resp.json().get("data", [])}


@app.get("/api/facebook/posts")
def list_facebook_posts():
    creds = get_valid_meta_credentials()
    if creds is None or not creds["pages"]:
        raise HTTPException(status_code=401, detail="Not connected")
    # Single-Page assumption for v1 - see plan's Milestone 3b.
    page = creds["pages"][0]

    resp = requests.get(
        f"{GRAPH_BASE}/{page['id']}/posts",
        params={
            "fields": "id,message,created_time,permalink_url",
            "access_token": page["access_token"],
        },
        timeout=10,
    )
    resp.raise_for_status()
    return {"posts": resp.json().get("data", [])}


@app.post("/api/monitor/{platform}/{content_id}/start")
async def monitor_start(platform: str, content_id: str):
    # async so this runs on the event loop thread - asyncio.create_task inside
    # start_monitor needs a running loop, which a threadpool-executed sync def
    # endpoint would not have.
    if platform not in PROCESSORS:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")
    started = start_monitor(platform, content_id)
    return {"monitoring": True, "already_running": not started}


@app.post("/api/monitor/{platform}/{content_id}/stop")
async def monitor_stop(platform: str, content_id: str):
    if platform not in PROCESSORS:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")
    stopped = stop_monitor(platform, content_id)
    return {"monitoring": False, "was_running": stopped}


@app.get("/api/monitor/{platform}/{content_id}/status")
def monitor_status(platform: str, content_id: str):
    if platform not in PROCESSORS:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")
    return {"monitoring": is_monitoring(platform, content_id)}


@app.get("/api/incidents/{platform}/{content_id}")
def incidents(platform: str, content_id: str):
    return {"incidents": get_incidents(content_id)}
