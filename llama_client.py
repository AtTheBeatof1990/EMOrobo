"""llama.cpp chat client using local llama-server or fallback to llama-cli."""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


class LlamaCppClient:
    """Chat client for local llama.cpp backend.

    Supports running `llama-server` as a managed local HTTP process (OpenAI-compatible)
    with seamless fallback to `llama-cli`.
    """

    def __init__(
        self,
        model_path: Path | str,
        server_bin: Path | str | None = None,
        cli_bin: Path | str | None = None,
        host: str = "http://127.0.0.1:8080",
        threads: int = 4,
        auto_start: bool = True,
        context_size: int = 2048,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.server_bin = Path(server_bin).resolve() if server_bin else None
        self.cli_bin = Path(cli_bin).resolve() if cli_bin else None
        self.host = (host or "http://127.0.0.1:8080").rstrip("/")
        self.threads = max(1, int(threads))
        self.auto_start = auto_start
        self.context_size = max(512, int(context_size))
        self._server_process: subprocess.Popen | None = None
        self._process_owner = False
        atexit.register(self.close)

    def is_server_alive(self, timeout: float = 1.0) -> bool:
        """Check if llama-server is responding to HTTP requests."""
        for endpoint in ("/health", "/v1/models"):
            url = f"{self.host}{endpoint}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "EMO-LlamaClient/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def ensure_server(self, timeout_seconds: float = 12.0) -> tuple[bool, str]:
        """Ensure llama-server is running and responsive."""
        if self.is_server_alive(timeout=0.8):
            return True, f"llama-server responding at {self.host}"

        if not self.auto_start:
            return False, f"llama-server is not running at {self.host} and auto_start is disabled"

        if not self.server_bin or not self.server_bin.is_file():
            # Search default locations if server_bin is not directly set
            candidates = [
                self.server_bin,
                self.model_path.parent.parent / "bin" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
                Path("bin/llama/llama-server.exe" if os.name == "nt" else "bin/llama/llama-server"),
            ]
            found = next((p for p in candidates if p and p.is_file()), None)
            if not found:
                return False, f"llama-server binary not found at {self.server_bin}"
            self.server_bin = found.resolve()

        if not self.model_path.is_file():
            return False, f"GGUF model file not found at {self.model_path}"

        # Parse host and port
        parsed = urllib.parse.urlparse(self.host)
        host_ip = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080

        cmd = [
            str(self.server_bin),
            "-m", str(self.model_path),
            "--host", str(host_ip),
            "--port", str(port),
            "-c", str(self.context_size),
            "-t", str(self.threads),
            "--log-disable",
        ]

        log.info("Starting llama-server: %s", " ".join(cmd))
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._process_owner = True
        except Exception as exc:
            return False, f"Failed to spawn llama-server: {exc}"

        # Wait for server to become responsive
        start = time.monotonic()
        while time.monotonic() - start < timeout_seconds:
            if self._server_process.poll() is not None:
                code = self._server_process.returncode
                self._server_process = None
                return False, f"llama-server exited unexpectedly with code {code}"
            if self.is_server_alive(timeout=0.6):
                log.info("llama-server is up and healthy at %s", self.host)
                return True, f"llama-server started successfully on {self.host}"
            time.sleep(0.3)

        return False, f"Timed out waiting for llama-server on {self.host}"

    def check(self) -> tuple[bool, str]:
        """Verify model and server readiness."""
        if not self.model_path.is_file():
            return False, f"Model file missing: {self.model_path}"

        model_name = self.model_path.stem
        # Check server first
        ok, detail = self.ensure_server()
        if ok:
            return True, f"llama.cpp ready ({model_name} via server)"

        # If server fails, check if CLI binary is available as fallback
        if self.cli_bin and self.cli_bin.is_file():
            return True, f"llama.cpp ready ({model_name} via CLI fallback)"

        return False, detail

    def _complete_via_server(self, messages: list[dict], temperature: float = 0.45, max_tokens: int = 160) -> str:
        url = f"{self.host}/v1/chat/completions"
        payload = json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "EMO-LlamaClient/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("llama-server returned no choices in response")
        content = (choices[0].get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("llama-server returned empty response content")
        return content

    def _complete_via_cli(self, prompt: str, max_tokens: int = 160) -> str:
        if not self.cli_bin or not self.cli_bin.is_file():
            raise RuntimeError("llama-cli binary not found")

        cmd = [
            str(self.cli_bin),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.threads),
            "--no-display-prompt",
        ]
        log.info("Running llama-cli: %s", " ".join(cmd[:4]))
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=40,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            raise RuntimeError(f"llama-cli failed with code {res.returncode}: {res.stderr[:200]}")
        output = res.stdout.strip()
        if not output:
            raise RuntimeError("llama-cli returned empty output")
        return output

    def complete(
        self,
        question: str,
        response_language: str = "en",
        history: list[dict] | None = None,
    ) -> str:
        """Query llama.cpp for an assistant completion."""
        language = response_language or "en"
        system = (
            "You are EMO, a friendly compact desktop robot assistant. Answer accurately in 1-4 short sentences. "
            f"Reply in language code {language}, unless the user explicitly asks for another language. "
            "Never claim you performed real-world actions unless a tool actually did so. "
            "When unsure, clearly say so. Keep answers short and spoken-word friendly."
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": (question or "")[:4000]})

        # Try server first
        try:
            if not self.is_server_alive(timeout=0.3):
                self.ensure_server()
            return self._complete_via_server(messages)[:5000]
        except Exception as server_exc:
            log.warning("llama-server query failed: %s; attempting CLI fallback...", server_exc)

        # Fallback to CLI
        try:
            cli_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            return self._complete_via_cli(cli_prompt)[:5000]
        except Exception as cli_exc:
            log.error("llama-cli fallback also failed: %s", cli_exc)
            raise RuntimeError(
                f"llama.cpp could not generate an answer (server: {server_exc}, cli: {cli_exc})"
            ) from server_exc

    def close(self) -> None:
        """Shut down the managed llama-server process if owned."""
        if self._process_owner and self._server_process is not None:
            log.info("Terminating owned llama-server process (PID %s)...", self._server_process.pid)
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=2.0)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            finally:
                self._server_process = None
                self._process_owner = False
