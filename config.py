"""Validated environment configuration for EMO's Raspberry Pi runtime."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)), 0)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _int(name, default)
    return value if minimum <= value <= maximum else default


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _float(name, default)
    return value if math.isfinite(value) and minimum <= value <= maximum else default


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _audio_device() -> str | int | None:
    value = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    if not value or value.casefold() in {"default", "auto", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _csv_ints(name: str, default: tuple[int, ...], minimum: int, maximum: int) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = tuple(int(part.strip(), 0) for part in raw.split(",") if part.strip())
        if not parsed or any(n < minimum or n > maximum for n in parsed):
            return default
        return parsed
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    base_dir: Path
    data_dir: Path
    cache_db: Path
    whisper_model: str
    whisper_translation_model: str
    whisper_download_root: Path
    whisper_backend: str
    whisper_cpp_bin: Path
    whisper_cpp_model: Path
    whisper_cpp_threads: int
    preload_whisper: bool
    sample_rate: int
    audio_input_device: str | int | None
    audio_rms_threshold: float
    max_utterance_seconds: float
    trailing_silence_seconds: float
    wake_word_required: bool
    wake_words: tuple[str, ...]
    wake_timeout_seconds: int
    vosk_model_path: Path
    llm_backend: str
    llama_cpp_bin: Path
    llama_cpp_cli_bin: Path
    llama_cpp_model: Path
    llama_cpp_host: str
    llama_cpp_threads: int
    weather_city: str
    time_zone: str
    cache_enabled: bool
    cache_ttl_seconds: int
    weather_cache_ttl_seconds: int
    tts_engine: str
    chatterbox_language_mode: str
    chatterbox_exaggeration: float
    chatterbox_cfg_weight: float
    chatterbox_temperature: float
    chatterbox_ref_audio: Path | None
    touch_gpio: int
    touch_power_gpio: int
    servo_gpios: tuple[int, int, int, int]
    mpu_addresses: tuple[int, ...]
    mpu_enabled: bool
    mpu_power_gpio: int
    i2c_bus: str
    i2c_sda_gpio: int
    i2c_scl_gpio: int
    remote_enabled: bool
    remote_host: str
    remote_port: int
    remote_token: str
    display_width: int
    display_height: int
    display_fullscreen: bool
    display_driver: str
    camera_enabled: bool
    camera_index: int
    gesture_enabled: bool
    face_recognition_enabled: bool
    object_classification_enabled: bool
    known_faces_dir: Path
    vision_model: str
    song_dir: Path
    ready_song: str
    thermal_warning_c: float
    thermal_lockout_c: float
    thermal_resume_c: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv(BASE_DIR / ".env")
        data_dir = _project_path(os.getenv("EMO_DATA_DIR", str(BASE_DIR / "data")))
        whisper_dir = _project_path(os.getenv(
            "WHISPER_DOWNLOAD_ROOT", str(Path.home() / ".cache" / "emo" / "whisper-en")
        ))
        wake_words = tuple(
            v.strip().casefold() for v in os.getenv(
                "WAKE_WORDS", "emo,hey emo,hi emo,hello emo,okay emo,ok emo"
            ).split(",") if v.strip()
        )
        if not wake_words:
            wake_words = ("emo", "hey emo", "hi emo", "hello emo")
        servos = _csv_ints("SERVO_GPIOS", (5, 6, 12, 13), 0, 27)
        if len(servos) != 4 or len(set(servos)) != 4:
            servos = (5, 6, 12, 13)
        addresses = _csv_ints("MPU6050_ADDRESSES", (0x68, 0x69), 0x03, 0x77)
        if len(set(addresses)) != len(addresses):
            addresses = (0x68, 0x69)
        warning = _bounded_float("THERMAL_WARNING_C", 75.0, 30.0, 120.0)
        lockout = _bounded_float("THERMAL_LOCKOUT_C", 82.0, 30.0, 120.0)
        resume = _bounded_float("THERMAL_RESUME_C", 72.0, 30.0, 120.0)
        if not resume < warning < lockout:
            warning, lockout, resume = 75.0, 82.0, 72.0
        ref_audio = os.getenv("CHATTERBOX_REF_AUDIO", "").strip()
        return cls(
            base_dir=BASE_DIR,
            data_dir=data_dir,
            cache_db=_project_path(os.getenv("EMO_CACHE_DB", str(data_dir / "emo_cache.sqlite3"))),
            whisper_model=os.getenv("WHISPER_MODEL", "tiny.en").strip().lower(),
            whisper_translation_model=os.getenv("WHISPER_TRANSLATION_MODEL", "tiny.en").strip().lower(),
            whisper_download_root=whisper_dir,
            whisper_backend=os.getenv("WHISPER_BACKEND", "faster-whisper").strip().lower(),
            whisper_cpp_bin=_project_path(os.getenv("WHISPER_CPP_BIN", str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"))),
            whisper_cpp_model=_project_path(os.getenv("WHISPER_CPP_MODEL", str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.en.bin"))),
            whisper_cpp_threads=_bounded_int("WHISPER_CPP_THREADS", 2, 1, 8),
            preload_whisper=_bool("PRELOAD_WHISPER", False),
            sample_rate=_bounded_int("AUDIO_SAMPLE_RATE", 16000, 8000, 48000),
            audio_input_device=_audio_device(),
            audio_rms_threshold=_bounded_float("AUDIO_RMS_THRESHOLD", 0.012, 0.0001, 0.5),
            max_utterance_seconds=_bounded_float("MAX_UTTERANCE_SECONDS", 8.0, 0.5, 30.0),
            trailing_silence_seconds=_bounded_float("TRAILING_SILENCE_SECONDS", 0.75, 0.1, 5.0),
            wake_word_required=_bool("WAKE_WORD_REQUIRED", True),
            wake_words=wake_words,
            wake_timeout_seconds=_bounded_int("WAKE_TIMEOUT_SECONDS", 20, 1, 300),
            vosk_model_path=_project_path(os.getenv(
                "VOSK_MODEL_PATH", str(Path.home() / ".cache" / "emo" / "vosk-model-small-en-us-0.15")
            )),
            llm_backend=os.getenv("LLM_BACKEND", "llama.cpp").strip().lower(),
            llama_cpp_bin=_project_path(os.getenv(
                "LLAMA_CPP_BIN",
                str(BASE_DIR / "bin" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")),
            )),
            llama_cpp_cli_bin=_project_path(os.getenv(
                "LLAMA_CPP_CLI_BIN",
                str(BASE_DIR / "bin" / "llama" / ("llama-cli.exe" if os.name == "nt" else "llama-cli")),
            )),
            llama_cpp_model=_project_path(os.getenv(
                "LLAMA_CPP_MODEL",
                str(data_dir / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            )),
            llama_cpp_host=os.getenv("LLAMA_CPP_HOST", "http://127.0.0.1:8080").rstrip("/"),
            llama_cpp_threads=_bounded_int("LLAMA_CPP_THREADS", 4, 1, 32),
            weather_city=os.getenv("WEATHER_CITY", "").strip(),
            time_zone=os.getenv("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata",
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_bounded_int("CACHE_TTL_SECONDS", 86400, 0, 31_536_000),
            weather_cache_ttl_seconds=_bounded_int("WEATHER_CACHE_TTL_SECONDS", 300, 0, 86_400),
            tts_engine=os.getenv("TTS_ENGINE", "auto").strip().lower(),
            chatterbox_language_mode=os.getenv("CHATTERBOX_LANGUAGE_MODE", "english").strip().lower(),
            chatterbox_exaggeration=_bounded_float("CHATTERBOX_EXAGGERATION", 0.5, 0.0, 1.5),
            chatterbox_cfg_weight=_bounded_float("CHATTERBOX_CFG_WEIGHT", 0.5, 0.0, 1.0),
            chatterbox_temperature=_bounded_float("CHATTERBOX_TEMPERATURE", 0.8, 0.1, 2.0),
            chatterbox_ref_audio=_project_path(ref_audio) if ref_audio else None,
            touch_gpio=_bounded_int("TOUCH_GPIO", 16, 0, 27),
            touch_power_gpio=_bounded_int("TOUCH_POWER_GPIO", 19, 0, 27),
            servo_gpios=servos,
            mpu_addresses=addresses,
            mpu_enabled=_bool("MPU6050_ENABLED", True),
            mpu_power_gpio=_bounded_int("MPU_POWER_GPIO", 26, 0, 27),
            i2c_bus=os.getenv("I2C_BUS", "auto").strip().lower() or "auto",
            i2c_sda_gpio=_bounded_int("I2C_SDA_GPIO", 20, 0, 27),
            i2c_scl_gpio=_bounded_int("I2C_SCL_GPIO", 21, 0, 27),
            remote_enabled=_bool("REMOTE_ENABLED", True),
            remote_host=os.getenv("REMOTE_HOST", "0.0.0.0").strip() or "0.0.0.0",
            remote_port=_bounded_int("REMOTE_PORT", 8766, 1024, 65535),
            remote_token=os.getenv("REMOTE_CONTROL_TOKEN", "").strip(),
            display_width=_bounded_int("DISPLAY_WIDTH", 320, 160, 1920),
            display_height=_bounded_int("DISPLAY_HEIGHT", 240, 120, 1080),
            display_fullscreen=_bool("DISPLAY_FULLSCREEN", True),
            display_driver=os.getenv("DISPLAY_DRIVER", "auto").strip().lower(),
            camera_enabled=_bool("CAMERA_ENABLED", False),
            camera_index=_bounded_int("CAMERA_INDEX", 0, 0, 16),
            gesture_enabled=_bool("GESTURE_ENABLED", False),
            face_recognition_enabled=_bool("FACE_RECOGNITION_ENABLED", False),
            object_classification_enabled=_bool("OBJECT_CLASSIFICATION_ENABLED", False),
            known_faces_dir=_project_path(os.getenv("KNOWN_FACES_DIR", str(data_dir / "known_faces"))),
            vision_model=os.getenv("VISION_MODEL", "yolo11n.pt").strip() or "yolo11n.pt",
            song_dir=_project_path(os.getenv("EMO_SONG_DIR", str(BASE_DIR / "assets" / "songs"))),
            ready_song=os.getenv("EMO_READY_SONG", "emo_is_ready.mp3").strip() or "emo_is_ready.mp3",
            thermal_warning_c=warning,
            thermal_lockout_c=lockout,
            thermal_resume_c=resume,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
