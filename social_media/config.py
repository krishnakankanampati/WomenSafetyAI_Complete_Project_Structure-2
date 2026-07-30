import os


def _load_dotenv():
    """
    Load KEY=value pairs from the project-root .env into the environment,
    for any key not already set. Mirrors ai_model/config.py's loader so
    social_media doesn't need to import from ai_model.
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

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
