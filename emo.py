#!/usr/bin/env python3
"""Pygame-first local EMO robot runtime."""
from __future__ import annotations
import logging
import os
import re
import signal
import threading

from assistant import EMOAssistant
from audio_io import AudioRecorder
from config import Config
from hardware import HardwareController, ThermalMonitor
from imu import MPU6050Array
from motion_engine import MOTIONS, MotionEngine
from llama_client import LlamaCppClient
from model_manager import ensure_all_models
from pygame_face import PygameFace
from remote_controller import RemoteController
from song_library import SongLibrary
from speech import WhisperRecognizer
from state import RobotStatus
from tts_engine import SpeechEngine
from translation import OfflineTranslator
from wake_word import WakeWordDetector
from vision.runtime import VisionRuntime

log = logging.getLogger("emo")


def strip_wake_word(text: str, phrases: tuple[str, ...]) -> tuple[bool, str]:
    clean = text.strip()
    folded = clean.casefold()
    for phrase in sorted(phrases, key=len, reverse=True):
        match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", folded)
        if match:
            remaining = (clean[:match.start()] + " " + clean[match.end():]).strip()
            return True, remaining.lstrip(" ,;:").strip()
    words = clean.split()
    if len(words) >= 2 and words[0].casefold() in {"hey", "hi", "hello"} and words[1].strip(",.!?:;").casefold() in {"emo", "emu"}:
        return True, " ".join(words[2:]).lstrip(" ,;:").strip()
    return False, clean


class EMOApplication:
    def __init__(self) -> None:
        self.config = Config.from_env()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        self.status = RobotStatus()
        self.stop_event = threading.Event()
        self.touch_event = threading.Event()
        self.face = PygameFace(
            self.status, self.config.display_width, self.config.display_height,
            self.config.display_fullscreen, self.config.display_driver,
            on_action=self.touch_event.set,
        )
        self.translator = OfflineTranslator()
        self.recorder = AudioRecorder(
            self.config.sample_rate, self.config.audio_rms_threshold,
            self.config.max_utterance_seconds, self.config.trailing_silence_seconds,
            status=self.status, input_device=self.config.audio_input_device,
            stop_event=self.stop_event,
        )
        self.recognizer = WhisperRecognizer(
            self.config.whisper_model, self.config.whisper_translation_model,
            self.config.whisper_download_root, self.translator, status=self.status,
            backend=self.config.whisper_backend,
            cpp_binary=self.config.whisper_cpp_bin,
            cpp_model=self.config.whisper_cpp_model,
            cpp_threads=self.config.whisper_cpp_threads,
        )
        self.wake = WakeWordDetector(
            self.config.vosk_model_path, self.config.wake_words,
            self.config.sample_rate, self.config.audio_input_device,
        )
        self.llm = LlamaCppClient(
            model_path=self.config.llama_cpp_model,
            server_bin=self.config.llama_cpp_bin,
            cli_bin=self.config.llama_cpp_cli_bin,
            host=self.config.llama_cpp_host,
            threads=self.config.llama_cpp_threads,
        )
        self.speech = SpeechEngine(
            self.config.tts_engine, status=self.status,
            exaggeration=self.config.chatterbox_exaggeration,
            cfg_weight=self.config.chatterbox_cfg_weight,
            temperature=self.config.chatterbox_temperature,
            reference_audio=self.config.chatterbox_ref_audio,
        )
        self.songs = SongLibrary(self.config.song_dir)
        self.hardware = HardwareController(
            self.config.touch_gpio, self.config.servo_gpios, self.status,
            on_touch=self.touch_event.set,
        )
        self.imu = MPU6050Array(self.config.mpu_addresses, status=self.status) if self.config.mpu_enabled else None
        self.thermal = ThermalMonitor(
            self.status, self.config.thermal_warning_c,
            self.config.thermal_lockout_c, self.config.thermal_resume_c,
        )
        self.motion = MotionEngine(self.hardware, self.stop_event)
        self.assistant = EMOAssistant(
            self.config, self.status, self.face, self.hardware,
            self.recognizer, self.translator, self.llm, self.speech,
            song_library=self.songs,
        )
        self.remote = RemoteController(
            self.status, self._remote_command, self.config.remote_host,
            self.config.remote_port, self.config.remote_token,
        )
        self.vision = VisionRuntime(self.config, self.status, self._on_gesture)
        self._interaction_thread = None

    def boot(self) -> None:
        self.face.start()
        self.status.set_boot_step("Configuration", "ok", "Loaded .env; Pygame display active")
        self.status.set_stage("Boot sequence", "Verifying models and starting local engines")
        try:
            ensure_all_models(self.config, status=self.status)
        except Exception as exc:
            log.warning("Automatic model check/download warning: %s", exc)
        try:
            from response_cache import ResponseCache
            cache = ResponseCache(self.config.cache_db, enabled=self.config.cache_enabled)
            cache.close()
            self.status.set_boot_step("Local cache", "ok", str(self.config.cache_db))
        except Exception as exc:
            self.status.set_boot_step("Local cache", "warning", str(exc))
        mic_ok, mic_detail = self.recorder.check_device()
        self.status.set_boot_step("Microphone", "ok" if mic_ok else "warning", mic_detail)
        wake_ok, wake_detail = self.wake.check()
        self.status.set_boot_step("Wake word", "ok" if wake_ok else "warning", wake_detail)
        whisper_ok, whisper_detail = self.recognizer.dependency_check()
        if whisper_ok and self.config.preload_whisper:
            try:
                self.recognizer.preload()
                whisper_detail = f"Preloaded {self.config.whisper_model} (CPU int8)"
            except Exception as exc:
                whisper_ok = False
                whisper_detail = f"Preload failed: {exc}"
        self.status.set_boot_step("Whisper", "ok" if whisper_ok else "warning", whisper_detail)
        llm_ok, llm_detail = self.llm.check()
        self.status.set_boot_step("LLM", "ok" if llm_ok else "warning", llm_detail)
        tts_ok, tts_detail = self.speech.check()
        self.status.set_boot_step("Speech output", "ok" if tts_ok else "warning", tts_detail)
        hw_ok, hw_detail = self.hardware.start()
        self.status.set_boot_step("Hardware", "ok" if hw_ok else "warning", hw_detail)
        if self.imu is not None:
            imu_ok, imu_detail = self.imu.start()
            self.status.set_boot_step("IMU", "ok" if imu_ok else "warning", imu_detail)
        else:
            self.status.set_boot_step("IMU", "warning", "MPU6050 disabled in .env")
        if self.config.remote_enabled:
            remote_ok, remote_detail = self.remote.start()
        else:
            remote_ok, remote_detail = False, "Remote controller disabled in .env"
        self.status.set_boot_step("Remote controller", "ok" if remote_ok else "warning", remote_detail)
        vision_ok, vision_detail = self.vision.start()
        self.status.set_boot_step("Vision", "ok" if vision_ok else "warning", vision_detail)
        self.thermal.start()
        self.motion.start()
        with self.status.lock:
            self.status.boot_complete = True
        self.face.set_state("idle")
        self.status.set_stage("Ready", "Say EMO, wait for the listening prompt, then say your request")
        log.info("EMO startup checks finished")
        threading.Thread(target=self._announce_ready, daemon=True, name="EMO-Ready-Announcement").start()

    def _announce_ready(self) -> None:
        if not self.songs.play_ready(self.config.ready_song):
            self.speech.speak("EMO is ready.", "en")

    def _on_gesture(self, gesture: str) -> None:
        mapping = {"wave": "A", "open": "A", "point": "C", "two": "J",
                   "three": "F", "four": "H", "stop": "P"}
        key = mapping.get((gesture or "").casefold())
        if key:
            self.motion.run_async(key)
            self.status.set_stage("Gesture detected", f"{gesture} → movement {key}")

    def _remote_command(self, command: str, argument: str) -> str | bool:
        if command in {"wake", "listen"}:
            self.touch_event.set()
            return "Wake requested; speak your request after the listening prompt"
        if command in {"sleep", "idle"}:
            self.face.set_state("idle")
            self.hardware.set_expression("idle")
            return "Idle pose selected"
        if command in {"motion", "move"}:
            key = (argument or "").strip().upper()
            if key in MOTIONS and self.motion.run_async(key):
                return f"Queued movement {key}: {MOTIONS[key][0]}"
            return False
        if command in {"say", "speak"}:
            text = " ".join((argument or "").split())
            if not text or len(text) > 300:
                return False
            threading.Thread(
                target=self.assistant._speak, args=(text, "en"),
                kwargs={"expression": "talking"}, daemon=True, name="EMO-Remote-Speech",
            ).start()
            return "Queued spoken message"
        if command == "sing":
            if not self.songs.songs():
                return False
            threading.Thread(
                target=self.songs.play_random, daemon=True, name="EMO-Remote-Song"
            ).start()
            return "Queued song playback"
        return False

    def _handle_transcription(self, path: str) -> None:
        if self.status.snapshot()["thermal_lockout"]:
            self.status.set_stage("Cooling down", "Heavy inference is paused until the Pi cools")
            return
        self.status.set_stage("Recognizing speech", f"Transcribing with {self.config.whisper_model}")
        try:
            result = self.recognizer.transcribe(path)
        finally:
            self.recognizer.release()
        text = (result.text or "").strip()
        if not text:
            self.status.set_stage("Wake standby", "No request recorded; say EMO again")
            return
        motion_match = re.fullmatch(
            r"(?:do\s+|move\s+|movement\s+|perform\s+)?([a-z])",
            text, flags=re.IGNORECASE,
        )
        if motion_match:
            key = motion_match.group(1).upper()
            if self.motion.run_async(key):
                self.assistant._speak(f"Okay, movement {key}: {MOTIONS[key][0]}.", "en", expression="excited")
                return
        normalized = re.sub(r"[.!?,;:]+$", "", text).strip().casefold()
        if normalized in {"sing", "sing a song", "play a song", "play my song"} and self.songs.play_random():
            self.assistant._speak("Here is one of my songs.", "en", expression="dance")
            return
        self.status.set_stage("Processing request", "Running local instructions or LLM chat")
        self.assistant.process(
            text, result.language or "en", result.english_text or text,
            result.translation_method or "faster-whisper",
        )

    def _interaction_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if self.status.snapshot()["thermal_lockout"]:
                    self.face.set_state("overheat")
                    self.status.set_stage("Cooling down", "Heavy inference is paused until the Pi cools")
                    self.stop_event.wait(2)
                    continue
                self.face.set_state("idle")
                self.status.set_stage("Wake standby", "Say EMO to activate Whisper, or tap the touch sensor")
                woke = self.wake.wait_for_wake(self.stop_event, self.touch_event)
                if not woke or self.stop_event.is_set():
                    continue
                self.face.set_state("listening")
                self.hardware.set_expression("listening")
                self.status.set_stage("Awake", "Whisper is enabled for your next request")
                self.speech.speak("I'm listening.", "en")
                self.status.set_stage("Listening for request", "Speak clearly now")
                path = self.recorder.record_utterance()
                if not path:
                    self.status.set_stage("Wake standby", "No request captured; say EMO again")
                    continue
                try:
                    self._handle_transcription(path)
                finally:
                    try: os.unlink(path)
                    except OSError: pass
                self.face.set_state("idle")
                self.hardware.set_expression("idle")
                self.status.set_stage("Wake standby", "Say EMO to start the next request")
            except Exception as exc:
                log.exception("Interaction cycle failed")
                with self.status.lock:
                    self.status.last_error = str(exc)[:240]
                self.status.set_stage("Recovering", str(exc)[:150])
                self.face.set_state("idle")
                self.stop_event.wait(1.5)

    def run(self) -> None:
        self.boot()
        self._interaction_thread = threading.Thread(target=self._interaction_loop, daemon=True, name="EMO-Interaction")
        self._interaction_thread.start()
        try:
            while not self.stop_event.is_set():
                if not self.face.update():
                    self.stop_event.set()
                    break
        finally:
            self.close()

    def close(self) -> None:
        self.stop_event.set()
        for action in (self.remote.stop, self.vision.stop, self.thermal.stop):
            try: action()
            except Exception: pass
        if self.imu is not None:
            try: self.imu.stop()
            except Exception: pass
        try: self.motion.close()
        except Exception: pass
        try: self.assistant.close()
        except Exception: pass
        if hasattr(self, "llm") and hasattr(self.llm, "close"):
            try: self.llm.close()
            except Exception: pass
        try: self.hardware.close()
        except Exception: pass
        try: self.face.close()
        except Exception: pass
        if self._interaction_thread and self._interaction_thread is not threading.current_thread():
            self._interaction_thread.join(timeout=2)


def main() -> int:
    app = EMOApplication()
    def stop_handler(_signum, _frame):
        app.stop_event.set()
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        app.close()
        return 0
    except Exception:
        logging.exception("Could not start EMO")
        app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
