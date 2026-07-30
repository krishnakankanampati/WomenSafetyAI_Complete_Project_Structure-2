"""
Builds the Google OAuth Flow and persists/reloads the resulting credentials
via MongoDB (database.mongodb.save_oauth_credentials/load_oauth_credentials),
so login survives a uvicorn --reload restart without needing a token file.
"""

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from backend.config import (
    GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET,
    GOOGLE_OAUTH_REDIRECT_URI,
    OAUTH_SCOPES,
)
from database.mongodb import load_oauth_credentials, save_oauth_credentials


def build_flow(code_verifier: str = None) -> Flow:
    client_config = {
        "web": {
            "client_id": GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_OAUTH_REDIRECT_URI],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=OAUTH_SCOPES, code_verifier=code_verifier)
    flow.redirect_uri = GOOGLE_OAUTH_REDIRECT_URI
    return flow


# google-auth-oauthlib auto-generates a PKCE code_verifier per Flow instance
# (autogenerate_code_verifier=True by default). /auth/login and /auth/callback
# are separate requests/Flow objects, so the verifier has to be handed off
# between them explicitly - single pending login at a time is fine for this
# single-user local tool.
_pending_code_verifier = None


def set_pending_verifier(verifier: str) -> None:
    global _pending_code_verifier
    _pending_code_verifier = verifier


def pop_pending_verifier():
    global _pending_code_verifier
    verifier, _pending_code_verifier = _pending_code_verifier, None
    return verifier


def _credentials_to_doc(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }


def save_credentials(creds: Credentials) -> None:
    save_oauth_credentials(_credentials_to_doc(creds))


def get_valid_credentials():
    """Returns valid Credentials, refreshing if expired, or None if never logged in."""
    doc = load_oauth_credentials()
    if not doc or not doc.get("refresh_token"):
        return None

    creds = Credentials(
        token=doc.get("token"),
        refresh_token=doc.get("refresh_token"),
        token_uri=doc.get("token_uri"),
        client_id=doc.get("client_id"),
        client_secret=doc.get("client_secret"),
        scopes=doc.get("scopes"),
    )

    if not creds.valid:
        creds.refresh(Request())
        save_credentials(creds)

    return creds
