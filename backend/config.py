import os


def _load_dotenv():
    """
    Load KEY=value pairs from the project-root .env into the environment,
    for any key not already set. Mirrors ai_model/config.py's loader so
    backend doesn't need to import from ai_model.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env",
    )
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


_load_dotenv()

# google-auth-oauthlib refuses a plain-http localhost redirect URI, and
# rejects Google's occasional habit of echoing extra default scopes back
# alongside the one requested - both need to be set before any Flow is built.
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

OAUTH_SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

META_APP_ID = os.environ.get("META_APP_ID")
META_APP_SECRET = os.environ.get("META_APP_SECRET")
META_OAUTH_REDIRECT_URI = os.environ.get("META_OAUTH_REDIRECT_URI", "http://localhost:8000/auth/meta/callback")

# Set this and the browser OAuth flow is bypassed entirely - see
# backend/meta_oauth_store._credentials_from_env_token for why that's the
# supported path for Meta here rather than a fallback.
META_PAGE_ACCESS_TOKEN = os.environ.get("META_PAGE_ACCESS_TOKEN")

# Verified live against the Meta console (2026-07-29): pages_read_engagement
# covers reading a Page's own posts, but the comments *edge* additionally
# needs pages_read_user_content - confirmed via Graph API Explorer testing,
# it 400s without it even with pages_read_engagement present. The two
# instagram_business_* scopes are the current names; instagram_basic /
# instagram_manage_comments (the original plan's guess) are retired.
META_OAUTH_SCOPES = (
    "instagram_business_basic,instagram_business_manage_comments,"
    "pages_show_list,pages_read_engagement,pages_read_user_content"
)
