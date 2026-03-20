from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Sequence


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _spawn(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(command)


def main() -> int:
    http_port = _env("GATEWAY_HTTP_PORT", "8000")
    https_port = _env("GATEWAY_HTTPS_PORT", "443")
    tls_cert_file = _env("TLS_CERT_FILE", "/certs/server.crt")
    tls_key_file = _env("TLS_KEY_FILE", "/certs/server.key")

    if not os.path.isfile(tls_cert_file) or not os.path.isfile(tls_key_file):
        print(
            f"TLS certificate files not found: {tls_cert_file} / {tls_key_file}",
            file=sys.stderr,
        )
        return 1

    processes = [
        _spawn(
            [
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                http_port,
            ]
        ),
        _spawn(
            [
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                https_port,
                "--ssl-keyfile",
                tls_key_file,
                "--ssl-certfile",
                tls_cert_file,
                "--proxy-headers",
            ]
        ),
    ]

    def terminate_children(*_: object) -> None:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)

    try:
        while True:
            for proc in processes:
                exit_code = proc.poll()
                if exit_code is not None:
                    terminate_children()
                    for child in processes:
                        if child is not proc:
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=10)
                    return exit_code
            time.sleep(0.5)
    finally:
        terminate_children()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
