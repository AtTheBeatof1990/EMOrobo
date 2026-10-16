"""Thread-safe runtime status shared by Pygame, hardware, and the remote controller."""
from __future__ import annotations
import threading
from datetime import datetime, timezone
from typing import Any


class RobotStatus:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.temperature = 0.0
        self.thermal_warning = False
        self.thermal_lockout = False
        self.camera_found = False
        self.last_gesture = ""
        self.imu_readings: dict[str, Any] = {}
        self.noise_floor = 0.0
        self.current_stage = "Starting EMO"
        self.current_message = "Starting local services"
        self.boot_complete = False
        self.last_action = ""
        self.last_error = ""
        self.last_touch = 0.0
        self.boot_started_at = datetime.now(timezone.utc).isoformat()
        self.boot_steps: dict[str, dict[str, str]] = {
            "Configuration": {"status": "pending", "detail": "Reading environment settings"},
            "Local cache": {"status": "pending", "detail": "Preparing SQLite cache"},
            "Microphone": {"status": "pending", "detail": "Waiting for audio device check"},
            "Whisper": {"status": "pending", "detail": "Model loads on first utterance"},
            "LLM": {"status": "pending", "detail": "Checking local LLM engine (llama.cpp)"},
            "Wake word": {"status": "pending", "detail": "Waiting for lightweight wake-word engine"},
            "IMU": {"status": "pending", "detail": "Checking MPU6050 sensors"},
            "Remote controller": {"status": "pending", "detail": "Waiting for remote-control token"},
            "Vision": {"status": "pending", "detail": "Optional camera stack is disabled"},
            "Speech output": {"status": "pending", "detail": "Checking available TTS engines"},
            "Hardware": {"status": "pending", "detail": "Optional GPIO sensors/servo"},
        }

    def set_stage(self, stage: str, message: str = "") -> None:
        with self.lock:
            self.current_stage = str(stage)[:80]
            if message:
                self.current_message = str(message)[:240]

    def set_boot_step(self, name: str, status: str, detail: str = "") -> None:
        with self.lock:
            self.boot_steps[name] = {
                "status": status if status in {"pending", "running", "ok", "warning", "error"} else "pending",
                "detail": str(detail)[:240],
            }

    def set_thermal(self, temperature: float, warning: bool, lockout: bool) -> None:
        with self.lock:
            self.temperature = float(temperature)
            self.thermal_warning = bool(warning)
            self.thermal_lockout = bool(lockout)

    def set_camera(self, found: bool, gesture: str = "") -> None:
        with self.lock:
            self.camera_found = bool(found)
            self.last_gesture = str(gesture)[:80]

    def set_imu(self, readings: dict[str, Any]) -> None:
        with self.lock:
            self.imu_readings = dict(readings)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "temperature": self.temperature,
                "thermal_warning": self.thermal_warning,
                "thermal_lockout": self.thermal_lockout,
                "camera_found": self.camera_found,
                "last_gesture": self.last_gesture,
                "imu": dict(self.imu_readings),
                "noise_floor": self.noise_floor,
                "current_stage": self.current_stage,
                "current_message": self.current_message,
                "boot_complete": self.boot_complete,
                "last_action": self.last_action,
                "last_error": self.last_error,
                "last_touch": self.last_touch,
                "boot_started_at": self.boot_started_at,
                "boot_steps": dict(self.boot_steps),
            }
