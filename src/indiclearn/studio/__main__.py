"""``python -m indiclearn.studio``  - run the web app (HOST / PORT env vars, defaults 127.0.0.1:8000)."""

import os


def main() -> None:
    import uvicorn

    uvicorn.run("indiclearn.studio.app:create_app", factory=True, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                log_level=os.environ.get("LOG_LEVEL", "info"), proxy_headers=True)


if __name__ == "__main__":
    main()
