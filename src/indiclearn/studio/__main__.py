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


def _port_busy(host: str, port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def _pids_on(port: int) -> list[int]:
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10).stdout
            return sorted({int(ln.split()[-1]) for ln in out.splitlines()
                           if "LISTENING" in ln and ln.split()[1].endswith(f":{port}") and ln.split()[-1].isdigit()} - {0})
        out = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=10).stdout
        return sorted({int(x) for x in out.split()})
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def free_port(host: str, port: int) -> bool:
    """An older IndicLearn still holding the port would keep serving old API code (pages say "Not Found"): stop it."""
    import json
    import signal
    import subprocess
    import sys
    import time
    import urllib.request

    if not _port_busy(host, port):
        return True
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
            ours = json.loads(r.read().decode()).get("ok") is True
    except (OSError, ValueError):
        ours = False
    if not ours:
        print(f"\nPort {port} is used by another program. Close it, or start with a different port: set PORT=8001\n", flush=True)
        return False
    print(f"An older IndicLearn server is still running on port {port} - stopping it so the new version starts.", flush=True)
    for pid in _pids_on(port):
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            pass
    for _ in range(20):
        if not _port_busy(host, port):
            return True
        time.sleep(0.5)
    print("\nCould not stop the old server. Close its window (or restart the computer) and run run.bat again.\n", flush=True)
    return False


def main() -> None:
    import sys

    import uvicorn

    load_dotenv()
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    if not free_port(host, port):
        sys.exit(1)
    uvicorn.run("indiclearn.studio.app:create_app", factory=True, host=host, port=port,
                log_level=os.environ.get("LOG_LEVEL", "info"), proxy_headers=True)


if __name__ == "__main__":
    main()
