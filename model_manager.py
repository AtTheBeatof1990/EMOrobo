"""Automatic verification and downloading for all EMO models (Vosk, Whisper, llama.cpp)."""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from config import Config

log = logging.getLogger(__name__)

VOSK_DEFAULT_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DEFAULT_NAME = "vosk-model-small-en-us-0.15"

LLAMA_CPP_WIN_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11681/llama-b11681-bin-win-cpu-x64.zip"
QWEN_GGUF_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"


class ModelManager:
    """Manages downloading, caching, and integrity validation for all required models."""

    def __init__(self, config: Config | None = None, log_fn: Callable[[str], None] | None = None) -> None:
        self.config = config or Config.from_env()
        self.log_fn = log_fn or log.info

    def _log(self, message: str) -> None:
        self.log_fn(message)

    # -------------------------------------------------------------------------
    # 1. Vosk Wake-Word Model
    # -------------------------------------------------------------------------
    def is_vosk_ready(self) -> bool:
        target = self.config.vosk_model_path
        return target.is_dir() and (target / "conf").is_dir() and (target / "am").is_dir()

    def ensure_vosk(self) -> tuple[bool, str]:
        if self.is_vosk_ready():
            return True, f"Vosk wake model already present at {self.config.vosk_model_path}"

        target = self.config.vosk_model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"Downloading Vosk wake-word model to {target}...")

        try:
            with tempfile.TemporaryDirectory(prefix="emo-vosk-dl-") as temp_dir:
                temp_path = Path(temp_dir).resolve()
                archive = temp_path / "model.zip"
                request = urllib.request.Request(
                    VOSK_DEFAULT_URL,
                    headers={"User-Agent": "EMO-ModelManager/1.0"},
                )
                with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)

                with zipfile.ZipFile(archive) as zipped:
                    for item in zipped.infolist():
                        resolved = (temp_path / item.filename).resolve()
                        if resolved != temp_path and temp_path not in resolved.parents:
                            raise RuntimeError(f"Unsafe archive entry: {item.filename}")
                    zipped.extractall(temp_path)

                extracted = temp_path / VOSK_DEFAULT_NAME
                if not (extracted / "conf").is_dir() or not (extracted / "am").is_dir():
                    raise RuntimeError("Extracted Vosk archive missing required model components.")

                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(extracted), str(target))

            self._log(f"Vosk wake model ready: {target}")
            return True, f"Downloaded Vosk wake model: {target.name}"
        except Exception as exc:
            msg = f"Failed to download Vosk wake model: {exc}"
            self._log(msg)
            return False, msg

    # -------------------------------------------------------------------------
    # 2. Faster-Whisper / Whisper Speech Model
    # -------------------------------------------------------------------------
    def is_whisper_ready(self) -> bool:
        cache_dir = self.config.whisper_download_root
        model_name = self.config.whisper_model
        if not cache_dir.is_dir():
            return False
        expected_dirs = [
            cache_dir / f"models--Systran--faster-whisper-{model_name}",
            cache_dir / f"faster-whisper-{model_name}",
            cache_dir / model_name,
        ]
        for ed in expected_dirs:
            if ed.is_dir():
                return True
        if self.config.whisper_backend == "cpp" and self.config.whisper_cpp_model.is_file():
            return True
        return False

    def ensure_whisper(self) -> tuple[bool, str]:
        if self.is_whisper_ready():
            return True, f"Whisper {self.config.whisper_model} already cached at {self.config.whisper_download_root}"

        model_name = self.config.whisper_model
        cache_dir = self.config.whisper_download_root
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Downloading/verifying Whisper model '{model_name}' into {cache_dir}...")

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(cache_dir),
            )
            del model
            self._log(f"Whisper model '{model_name}' is ready.")
            return True, f"Downloaded Whisper model: {model_name}"
        except Exception as exc:
            msg = f"Failed to download Whisper model '{model_name}': {exc}"
            self._log(msg)
            return False, msg

    # -------------------------------------------------------------------------
    # 3. llama.cpp Binaries
    # -------------------------------------------------------------------------
    def is_llama_cpp_bin_ready(self) -> bool:
        server_bin = self.config.llama_cpp_bin
        cli_bin = self.config.llama_cpp_cli_bin
        return server_bin.is_file() or cli_bin.is_file()

    def ensure_llama_cpp_binaries(self) -> tuple[bool, str]:
        if self.is_llama_cpp_bin_ready():
            return True, f"llama.cpp binaries already present at {self.config.llama_cpp_bin.parent}"

        if os.name != "nt":
            return False, "Automatic binary download is configured for Windows x64. Please compile or install llama.cpp manually."

        target_dir = self.config.llama_cpp_bin.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Downloading pre-built llama.cpp binaries into {target_dir}...")

        try:
            with tempfile.TemporaryDirectory(prefix="emo-llama-dl-") as temp_dir:
                temp_path = Path(temp_dir).resolve()
                archive = temp_path / "llama.zip"
                request = urllib.request.Request(
                    LLAMA_CPP_WIN_URL,
                    headers={"User-Agent": "EMO-ModelManager/1.0"},
                )
                with urllib.request.urlopen(request, timeout=300) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)

                with zipfile.ZipFile(archive) as zipped:
                    for item in zipped.infolist():
                        resolved = (temp_path / item.filename).resolve()
                        if resolved != temp_path and temp_path not in resolved.parents:
                            raise RuntimeError(f"Unsafe archive entry: {item.filename}")
                    zipped.extractall(target_dir)

            if not self.is_llama_cpp_bin_ready():
                raise RuntimeError("Extracted archive did not contain expected llama-server or llama-cli binaries.")

            self._log(f"llama.cpp binaries ready in {target_dir}")
            return True, f"Downloaded llama.cpp binaries into {target_dir.name}"
        except Exception as exc:
            msg = f"Failed to download llama.cpp binaries: {exc}"
            self._log(msg)
            return False, msg

    # -------------------------------------------------------------------------
    # 4. llama.cpp GGUF Model
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_valid_gguf(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            return False
        try:
            with open(path, "rb") as f:
                header = f.read(4)
                return header == b"GGUF"
        except Exception:
            return False

    def is_llama_cpp_model_ready(self) -> bool:
        return self._is_valid_gguf(self.config.llama_cpp_model)

    def _find_local_cached_gguf(self) -> Path | None:
        """Search local drive for existing compatible GGUF blob for Qwen 0.5B to avoid redownloading."""
        user_dirs = [Path.home(), Path("C:/Users/ACER"), Path("C:/Users/lakshya singh rana")]
        try:
            for p in Path("C:/Users").iterdir():
                if p.is_dir() and p not in user_dirs:
                    user_dirs.append(p)
        except Exception:
            pass

        for u in user_dirs:
            blobs_dir = u / ".ollama" / "models" / "blobs"
            if not blobs_dir.is_dir():
                continue
            for blob in blobs_dir.glob("sha256-*"):
                try:
                    size = blob.stat().st_size
                    if 350_000_000 <= size <= 450_000_000 and self._is_valid_gguf(blob):
                        return blob
                except Exception:
                    continue
        return None

    def ensure_llama_cpp_model(self) -> tuple[bool, str]:
        target = self.config.llama_cpp_model
        if self.is_llama_cpp_model_ready():
            return True, f"GGUF model already present at {target}"

        target.parent.mkdir(parents=True, exist_ok=True)

        # 1. Check local cache first
        local_blob = self._find_local_cached_gguf()
        if local_blob:
            self._log(f"Found local GGUF model ({local_blob.name}); copying to {target}...")
            try:
                shutil.copyfile(local_blob, target)
                if self.is_llama_cpp_model_ready():
                    self._log(f"Imported GGUF model from local cache: {target}")
                    return True, f"Imported local model: {target.name}"
            except Exception as exc:
                self._log(f"Local copy failed: {exc}; will download from HuggingFace")

        # 2. Download from HuggingFace
        self._log(f"Downloading Qwen2.5 0.5B GGUF model (~397MB) from HuggingFace into {target}...")
        try:
            temp_target = target.with_suffix(".download")
            request = urllib.request.Request(
                QWEN_GGUF_URL,
                headers={"User-Agent": "EMO-ModelManager/1.0"},
            )
            with urllib.request.urlopen(request, timeout=600) as response, temp_target.open("wb") as output:
                shutil.copyfileobj(response, output)

            if not self._is_valid_gguf(temp_target):
                temp_target.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is not a valid GGUF model.")

            if target.exists():
                target.unlink()
            temp_target.rename(target)

            self._log(f"GGUF model ready: {target}")
            return True, f"Downloaded GGUF model: {target.name}"
        except Exception as exc:
            msg = f"Failed to download GGUF model: {exc}"
            self._log(msg)
            return False, msg

    # -------------------------------------------------------------------------
    # 5. Master check & download all models
    # -------------------------------------------------------------------------
    def ensure_all(self, status=None) -> dict[str, tuple[bool, str]]:
        results = {}

        if status is not None:
            status.set_stage("Model check", "Checking and downloading required models...")

        # 1. Vosk
        if not self.is_vosk_ready():
            if status is not None:
                status.set_stage("Downloading model", "Downloading Vosk wake-word model...")
            results["vosk"] = self.ensure_vosk()
        else:
            results["vosk"] = (True, "Vosk wake model present")

        # 2. Whisper
        if not self.is_whisper_ready():
            if status is not None:
                status.set_stage("Downloading model", f"Downloading Whisper {self.config.whisper_model}...")
            results["whisper"] = self.ensure_whisper()
        else:
            results["whisper"] = (True, f"Whisper {self.config.whisper_model} present")

        # 3. LLM Backend
        if self.config.llm_backend == "llama.cpp":
            # llama.cpp binaries
            if not self.is_llama_cpp_bin_ready():
                if status is not None:
                    status.set_stage("Downloading tools", "Downloading llama.cpp binaries...")
                results["llama_bin"] = self.ensure_llama_cpp_binaries()
            else:
                results["llama_bin"] = (True, "llama.cpp binaries present")

            # llama.cpp model
            if not self.is_llama_cpp_model_ready():
                if status is not None:
                    status.set_stage("Downloading model", "Downloading Qwen2.5 0.5B GGUF model...")
                results["llama_model"] = self.ensure_llama_cpp_model()
            else:
                results["llama_model"] = (True, "llama.cpp GGUF model present")

        return results


def ensure_all_models(config: Config | None = None, status=None) -> dict[str, tuple[bool, str]]:
    manager = ModelManager(config)
    return manager.ensure_all(status=status)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="EMO Model Verification & Downloader")
    parser.add_argument("--check", action="store_true", help="Check status without downloading")
    parser.add_argument("--all", action="store_true", default=True, help="Download all missing models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=" * 60)
    print("           EMO Model Verification & Downloader")
    print("=" * 60)
    cfg = Config.from_env()
    manager = ModelManager(cfg, log_fn=print)

    if args.check:
        print(f" Vosk Ready:         {manager.is_vosk_ready()}")
        print(f" Whisper Ready:      {manager.is_whisper_ready()}")
        print(f" llama.cpp Binaries: {manager.is_llama_cpp_bin_ready()}")
        print(f" llama.cpp GGUF:     {manager.is_llama_cpp_model_ready()}")
        return 0

    results = manager.ensure_all()
    all_ok = True
    for key, (ok, detail) in results.items():
        status_label = "[OK]" if ok else "[FAILED]"
        print(f" {status_label} {key.upper()}: {detail}")
        if not ok:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("All models verified and ready!")
        return 0
    else:
        print("Some models could not be prepared. Check network connection or configuration.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
