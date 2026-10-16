#!/usr/bin/env python3
"""Configure .env with Windows or custom desktop defaults."""
from __future__ import annotations
import secrets
from pathlib import Path


def main() -> int:
    env_path = Path(".env")
    if not env_path.exists():
        example_path = Path(".env.example")
        if example_path.exists():
            env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            env_path.write_text("", encoding="utf-8")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    settings = {
        "DISPLAY_FULLSCREEN": "false",
        "DISPLAY_WIDTH": "640",
        "DISPLAY_HEIGHT": "480",
        "MPU6050_ENABLED": "false",
        "TTS_ENGINE": "auto",
        "WHISPER_BACKEND": "faster-whisper",
        "WHISPER_MODEL": "tiny.en",
        "PRELOAD_WHISPER": "true",
    }

    seen = set()
    updated = []
    has_token = False
    for line in lines:
        stripped = line.strip()
        if "=" in line and not stripped.startswith("#"):
            k, v = line.split("=", 1)
            key = k.strip()
            if key == "REMOTE_CONTROL_TOKEN":
                has_token = True
                if not v.strip():
                    updated.append(f"REMOTE_CONTROL_TOKEN={secrets.token_urlsafe(32)}")
                else:
                    updated.append(line)
                seen.add(key)
            elif key in settings:
                updated.append(f"{key}={settings[key]}")
                seen.add(key)
            else:
                updated.append(line)
        else:
            updated.append(line)

    for k, v in settings.items():
        if k not in seen:
            updated.append(f"{k}={v}")
    if not has_token:
        updated.append(f"REMOTE_CONTROL_TOKEN={secrets.token_urlsafe(32)}")

    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print("Configured .env with desktop settings and remote control token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
