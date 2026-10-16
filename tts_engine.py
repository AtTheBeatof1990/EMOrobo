"""Speech output with optional Chatterbox and a lightweight espeak-ng fallback."""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import wave

log = logging.getLogger(__name__)

_TTS_LOCK = threading.RLock()

ESPEAK_VOICES = {
    "en": "en-us", "hi": "hi", "pa": "pa", "es": "es", "fr": "fr",
    "de": "de", "it": "it", "pt": "pt", "ru": "ru", "zh": "zh",
    "ja": "ja", "ko": "ko", "ar": "ar", "ur": "ur",
}


class SpeechEngine:
    def __init__(self, engine: str = "auto", status=None, *, exaggeration: float = 0.5,
                 cfg_weight: float = 0.5, temperature: float = 0.8, reference_audio=None) -> None:
        self.engine = engine
        self.status = status
        self.exaggeration = min(1.5, max(0.0, float(exaggeration)))
        self.cfg_weight = min(1.0, max(0.0, float(cfg_weight)))
        self.temperature = min(2.0, max(0.1, float(temperature)))
        self.reference_audio = str(reference_audio) if reference_audio else None
        self._english_model = None
        self._multilingual_model = None
        self.last_engine = "none"
        self._chatterbox_failed = False
        self._player = shutil.which("aplay")
        self._memory_mb = self._total_memory_mb()

    @staticmethod
    def _total_memory_mb() -> int:
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                first = handle.readline()
            return int(first.split()[1]) // 1024
        except Exception:
            return 0

    @staticmethod
    def _pyttsx3_available() -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:
            return False

    def check(self) -> tuple[bool, str]:
        if self.engine == "pyttsx3":
            available = self._pyttsx3_available()
            return available, "pyttsx3 is available" if available else "Install pyttsx3"
        if self.engine == "espeak":
            available = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
            return available, "espeak-ng is available" if available else "Install espeak-ng"
        if self.engine in {"auto", "chatterbox"}:
            if self._memory_mb and self._memory_mb < 1800:
                if shutil.which("espeak-ng") or shutil.which("espeak"):
                    return True, f"Low-memory device ({self._memory_mb} MB); using espeak fallback to avoid loading Chatterbox"
                if self._pyttsx3_available():
                    return True, "Low-memory device; using pyttsx3 native speech engine"
            try:
                import chatterbox  # noqa: F401
                return True, "Chatterbox package found; model loads on first response"
            except Exception:
                if shutil.which("espeak-ng") or shutil.which("espeak"):
                    return True, "Chatterbox unavailable; espeak fallback available"
                if self._pyttsx3_available():
                    return True, "Chatterbox unavailable; pyttsx3 fallback available"
                return False, "No usable TTS engine installed"
        if self._pyttsx3_available():
            return True, "Using pyttsx3 speech engine"
        return bool(shutil.which("espeak-ng") or shutil.which("espeak")), "Using espeak fallback"

    def speak(self, text: str, language: str = "en", *, sensitive: bool = False) -> bool:
        text = " ".join((text or "").split())
        if not text:
            return False
        if self.engine == "pyttsx3":
            if self._speak_pyttsx3(text, language):
                self.last_engine = "pyttsx3"
                return True
        # On a Pi 3B+ with roughly 1 GB RAM, don't attempt to load a neural TTS model.
        can_try_chatterbox = self.engine in {"auto", "chatterbox"} and not self._chatterbox_failed
        if self._memory_mb and self._memory_mb < 1800:
            can_try_chatterbox = self.engine == "chatterbox"  # explicit opt-in only on low-RAM hardware
        if self.engine in {"espeak", "pyttsx3"}:
            can_try_chatterbox = False
        if can_try_chatterbox and self._speak_chatterbox(text, language):
            self.last_engine = "chatterbox"
            return True
        if self.engine != "pyttsx3" and self._speak_espeak(text, language):
            self.last_engine = "espeak-ng"
            return True
        if self.engine in {"auto", "pyttsx3"} and self._speak_pyttsx3(text, language):
            self.last_engine = "pyttsx3"
            return True
        self.last_engine = "console"
        if sensitive:
            print("EMO: [sensitive response omitted because no voice engine is available]")
        else:
            print(f"EMO: {text}")
        return False

    def _speak_pyttsx3(self, text: str, language: str = "en") -> bool:
        with _TTS_LOCK:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(text)
                if getattr(engine, "_inLoop", False):
                    try:
                        engine.endLoop()
                    except Exception:
                        pass
                engine.runAndWait()
                return True
            except Exception as exc:
                log.warning("pyttsx3 playback failed: %s", exc)
                return False

    def _speak_chatterbox(self, text: str, language: str) -> bool:
        try:
            import numpy as np
            import torch
            if language == "en":
                if self._english_model is None:
                    from chatterbox.tts import ChatterboxTTS
                    self._english_model = ChatterboxTTS.from_pretrained(device="cpu")
                model = self._english_model
                audio = model.generate(
                    text, exaggeration=self.exaggeration, cfg_weight=self.cfg_weight,
                    temperature=self.temperature, audio_prompt_path=self.reference_audio,
                )
            else:
                if self._multilingual_model is None:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                model = self._multilingual_model
                supported = model.get_supported_languages()
                if language not in supported:
                    log.info("Chatterbox has no configured language ID %s; using fallback voice.", language)
                    return False
                audio = model.generate(
                    text, language_id=language, exaggeration=self.exaggeration,
                    cfg_weight=self.cfg_weight, temperature=self.temperature,
                    audio_prompt_path=self.reference_audio,
                )
            data = audio.detach().cpu().float().squeeze().numpy()
            data = np.nan_to_num(data).clip(-1.0, 1.0)
            pcm = (data * 32767).astype("<i2", copy=False).tobytes()
            fd, path = tempfile.mkstemp(prefix="emo-voice-", suffix=".wav")
            os.close(fd)
            try:
                with wave.open(path, "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(int(model.sr))
                    output.writeframes(pcm)
                if self._player:
                    subprocess.run([self._player, "-q", path], check=True, timeout=90)
                    return True
                if self._play_file(path):
                    return True
                log.warning("Chatterbox audio generated, but no audio player is available.")
                return False
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        except Exception as exc:
            self._chatterbox_failed = True
            log.warning("Chatterbox TTS failed; switching to lightweight voice: %s", exc)
            return False

    @staticmethod
    def _play_file(path: str) -> bool:
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return True
        except Exception:
            pass
        try:
            import pygame.mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            sound = pygame.mixer.Sound(path)
            sound.play()
            import time
            while pygame.mixer.get_busy():
                time.sleep(0.05)
            return True
        except Exception:
            pass
        return False

    def _speak_espeak(self, text: str, language: str) -> bool:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            return False
        voice = ESPEAK_VOICES.get(language, "en-us")
        candidates = [voice] if voice == "en-us" else [voice, "en-us"]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [executable, "-v", candidate, "-s", "165", text],
                    check=False, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    if candidate != voice:
                        log.info("Voice %s is unavailable; using en-us fallback.", voice)
                    return True
                log.debug("espeak voice %s exited with status %s", candidate, result.returncode)
            except Exception as exc:
                log.warning("espeak playback failed for voice %s: %s", candidate, exc)
        return False
