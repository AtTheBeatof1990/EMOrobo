"""Local microphone capture and small synthesized animal/song effects."""
from __future__ import annotations
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import threading
import wave
import logging

log = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, sample_rate: int, threshold: float, max_seconds: float,
                 trailing_silence: float, status=None, input_device=None,
                 stop_event: threading.Event | None = None) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_seconds = max_seconds
        self.trailing_silence = trailing_silence
        self.status = status
        # None means auto-select: prefer the OS default input, then the first input device.
        # A configured integer selects a PortAudio device index; a string selects a device name.
        self.input_device = input_device
        self.stop_event = stop_event
        self._last_device_warning = 0.0

    def _resolve_input_device(self, sd):
        """Return (device argument, display name, error detail), avoiding an invalid default -1."""
        try:
            devices = list(sd.query_devices())
        except Exception as exc:
            return None, "", f"PortAudio could not enumerate audio devices: {exc}"
        inputs = [(index, item) for index, item in enumerate(devices)
                  if int(item.get("max_input_channels", 0) or 0) > 0]
        if self.input_device is not None:
            if isinstance(self.input_device, str):
                needle = self.input_device.strip().casefold()
                matches = [
                    (index, info) for index, info in inputs
                    if needle and needle in str(info.get("name", "")).casefold()
                ]
                if len(matches) == 1:
                    index, info = matches[0]
                    return index, str(info.get("name", index)), ""
                if len(matches) > 1:
                    names = ", ".join(f"{info.get('name', index)!r} (index {index})" for index, info in matches)
                    return None, "", f"Microphone name {self.input_device!r} is ambiguous; matches: {names}. Use its numeric index."
                return None, "", f"No input device name contains {self.input_device!r}. Check sounddevice.query_devices()."
            try:
                index = int(self.input_device)
                info = sd.query_devices(index)
            except Exception as exc:
                return None, "", f"Configured microphone {self.input_device!r} is unavailable: {exc}"
            if int(info.get("max_input_channels", 0) or 0) < 1:
                return None, "", f"Configured device {self.input_device!r} has no microphone input."
            return index, str(info.get("name", index)), ""

        try:
            defaults = sd.default.device
            default_input = int(defaults[0]) if defaults is not None else -1
        except Exception:
            default_input = -1
        if default_input >= 0:
            for index, info in inputs:
                if index == default_input:
                    return index, str(info.get("name", f"device {index}")), ""
        if inputs:
            index, info = inputs[0]
            return index, str(info.get("name", f"device {index}")), ""
        return None, "", "No microphone input device is available. Connect a USB microphone or configure a supported audio input."

    def _audio_unavailable(self, detail: str) -> None:
        now = time.monotonic()
        if now - self._last_device_warning >= 30:
            log.warning("Microphone unavailable: %s", detail)
            self._last_device_warning = now
        if self.status is not None:
            self.status.set_boot_step("Microphone", "warning", detail)
            self.status.set_stage("Microphone unavailable", detail)

    def _resolve_sample_rate(self, sd, device) -> int:
        """Prefer the configured rate; fall back to the hardware's default and resample."""
        try:
            sd.check_input_settings(
                device=device, channels=1, dtype="float32", samplerate=self.sample_rate
            )
            return self.sample_rate
        except Exception as requested_error:
            try:
                info = sd.query_devices(device)
                native_rate = int(round(float(info.get("default_samplerate", 0))))
                if native_rate <= 0:
                    raise RuntimeError("The device did not report a valid default sample rate.")
                if native_rate == self.sample_rate:
                    raise RuntimeError(f"The device reports {native_rate} Hz but rejected that rate.")
                sd.check_input_settings(
                    device=device, channels=1, dtype="float32", samplerate=native_rate
                )
                log.info(
                    "Microphone does not support %s Hz; capturing at %s Hz and resampling to %s Hz.",
                    self.sample_rate, native_rate, self.sample_rate,
                )
                return native_rate
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Microphone cannot open at {self.sample_rate} Hz and its default rate could not be used: "
                    f"{fallback_error}"
                ) from requested_error

    def check_device(self) -> tuple[bool, str]:
        try:
            import sounddevice as sd
            device, name, error = self._resolve_input_device(sd)
            if device is None:
                return False, error or "No microphone input device found"
            return True, f"Using input device {name!r} (index/device={device!r})"
        except Exception as exc:
            return False, f"Microphone check failed: {exc}"

    def record_utterance(self) -> str | None:
        """Wait for speech energy, record until silence, and return a temporary WAV path."""
        if self.stop_event is not None and self.stop_event.is_set():
            return None
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install numpy and sounddevice, and verify PortAudio is installed.") from exc

        device, device_name, device_error = self._resolve_input_device(sd)
        if device is None:
            self._audio_unavailable(device_error or "No microphone input device found.")
            time.sleep(2.0)
            return None

        try:
            capture_rate = self._resolve_sample_rate(sd, device)
        except Exception as exc:
            self._audio_unavailable(f"Could not select a supported sample rate for {device_name!r}: {exc}")
            if self.stop_event is None or not self.stop_event.is_set():
                time.sleep(2.0)
            return None

        block_frames = max(400, int(capture_rate * 0.10))
        chunks = []
        began = False
        voiced_seconds = 0.0
        silent_seconds = 0.0
        deadline = time.monotonic() + self.max_seconds + 12.0
        noise = 0.0
        try:
            with sd.InputStream(device=device, samplerate=capture_rate, channels=1,
                                dtype="float32", blocksize=block_frames) as stream:
                while time.monotonic() < deadline and not (
                    self.stop_event is not None and self.stop_event.is_set()
                ):
                    block, _overflowed = stream.read(block_frames)
                    if block is None or len(block) == 0:
                        continue
                    mono = block[:, 0].copy()
                    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
                    if not began:
                        noise = 0.96 * noise + 0.04 * rms
                    if self.status is not None:
                        with self.status.lock:
                            self.status.noise_floor = noise
                    if rms >= self.threshold:
                        began = True
                        chunks.append(mono)
                        voiced_seconds += len(mono) / capture_rate
                        silent_seconds = 0.0
                    elif began:
                        chunks.append(mono)
                        silent_seconds += len(mono) / capture_rate
                    if began and (silent_seconds >= self.trailing_silence or voiced_seconds >= self.max_seconds):
                        break
        except Exception as exc:
            self._audio_unavailable(
                f"Could not read microphone {device_name!r} at {capture_rate} Hz: {exc}. "
                "Check the device index, connection and audio permissions."
            )
            if self.stop_event is None or not self.stop_event.is_set():
                time.sleep(2.0)
            return None

        if self.stop_event is not None and self.stop_event.is_set():
            return None
        if not began or voiced_seconds < 0.18:
            return None
        audio = np.concatenate(chunks).clip(-1.0, 1.0)
        if capture_rate != self.sample_rate and audio.size > 1:
            target_count = max(1, int(round(audio.size * self.sample_rate / capture_rate)))
            target_positions = np.linspace(0, audio.size - 1, num=target_count)
            audio = np.interp(target_positions, np.arange(audio.size), audio).astype("float32")
        pcm = (audio * 32767).astype("<i2", copy=False).tobytes()
        fd, path = tempfile.mkstemp(prefix="emo-in-", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm)
        return path


class SoundEffects:
    """Generates toy-like sound approximations locally; these are not animal recordings."""
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        self.player = shutil.which("aplay")

    def play_animal(self, animal: str) -> bool:
        specs = {"cat": (520, 860, .85, 1.1), "dog": (180, 100, .65, .7),
                 "wolf": (230, 470, 1.6, 1.0), "fox": (720, 1050, .35, .8)}
        start, end, duration, gain = specs.get(animal, specs["dog"])
        segments = 3 if animal == "dog" else 2 if animal in {"fox", "cat"} else 1
        samples = []
        total = int(self.SAMPLE_RATE * duration)
        for index in range(total):
            t = index / self.SAMPLE_RATE
            fraction = index / max(1, total - 1)
            freq = start + (end - start) * fraction
            if segments > 1:
                phase = (t * segments / duration) % 1.0
                envelope = min(1.0, phase * 9.0) * min(1.0, (1.0 - phase) * 8.0)
            else:
                envelope = min(1.0, t / .12) * min(1.0, (duration - t) / .16)
            value = (math.sin(2 * math.pi * freq * t) + .28 * math.sin(2 * math.pi * freq * 2.1 * t)) * envelope * gain
            samples.append(int(max(-1.0, min(1.0, value)) * 24000))
        return self._play_samples(samples)

    def play_original_tune(self) -> bool:
        notes = [(523.25, .18), (587.33, .18), (659.25, .22), (783.99, .28),
                 (659.25, .18), (783.99, .32), (880.00, .38)]
        samples = []
        sr = self.SAMPLE_RATE
        for frequency, duration in notes:
            count = int(sr * duration)
            for i in range(count):
                edge = min(1.0, i / (sr * .02), (count - i) / (sr * .04))
                samples.append(int(math.sin(2 * math.pi * frequency * i / sr) * max(0.0, edge) * .20 * 32767))
            samples.extend([0] * int(sr * .025))
        return self._play_samples(samples)

    def _play_samples(self, samples: list[int]) -> bool:
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="emo-fx-", suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self.SAMPLE_RATE)
                output.writeframes(struct.pack("<" + "h" * len(samples), *samples))
            if self.player:
                result = subprocess.run([self.player, "-q", path], check=False, timeout=6,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode != 0:
                    log.debug("aplay exited with status %s", result.returncode)
                return result.returncode == 0
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
                while pygame.mixer.get_busy():
                    time.sleep(0.05)
                return True
            except Exception:
                pass
            return False
        except Exception as exc:
            log.debug("Effect playback failed: %s", exc)
            return False
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
