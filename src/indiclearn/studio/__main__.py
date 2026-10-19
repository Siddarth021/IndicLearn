"""``python -m indiclearn.studio``  - run the web app (HOST / PORT env vars, defaults 127.0.0.1:8000)."""

import os


def load_dotenv(path: str = ".env") -> int:
    """Minimal .env reader (KEY=VALUE lines, # comments). Real environment variables win."""
    n = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split(" #", 1)[0].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    n += 1
    except OSError:
        pass
    return n


def main() -> None:
    import uvicorn

    load_dotenv()
    uvicorn.run("indiclearn.studio.app:create_app", factory=True, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                log_level=os.environ.get("LOG_LEVEL", "info"), proxy_headers=True)


if __name__ == "__main__":
    main()
