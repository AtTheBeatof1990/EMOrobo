"""Play user-provided EMO MP3s without shipping copyrighted music."""
from __future__ import annotations

import logging
import random
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class SongLibrary:
    def __init__(self, song_dir: Path) -> None:
        self.song_dir = Path(song_dir)
        self.song_dir.mkdir(parents=True, exist_ok=True)

    def songs(self) -> list[Path]:
        return sorted(
            (path for path in self.song_dir.glob("*")
             if path.is_file() and path.suffix.casefold() in {".mp3", ".wav", ".ogg"}),
            key=lambda p: p.name.casefold(),
        )

    def play(self, name: str) -> bool:
        target = Path(name)
        if not target.is_absolute():
            target = self.song_dir / target
        if not target.is_file() or target.suffix.casefold() not in {".mp3", ".wav", ".ogg"}:
            return False
        player = shutil.which("mpg123")
        command = [player, "-q", str(target)] if player else None
        if command is None and shutil.which("ffplay"):
            command = [shutil.which("ffplay"), "-nodisp", "-autoexit", "-loglevel", "error", str(target)]
        if command is None and shutil.which("cvlc"):
            command = [shutil.which("cvlc"), "--play-and-exit", "--quiet", str(target)]
        if command is None:
            try:
                import pygame.mixer
                import time
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(str(target))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                return True
            except Exception as exc:
                log.warning("pygame.mixer playback failed for %s: %s", target.name, exc)
                return False
        try:
            subprocess.run(command, check=True, timeout=180)
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("Could not play song %s: %s", target.name, exc)
            return False

    def play_random(self) -> bool:
        tracks = self.songs()
        if not tracks:
            log.info("No EMO songs installed yet; add MP3s to %s", self.song_dir)
            return False
        return self.play(random.choice(tracks).name)

    def play_ready(self, filename: str) -> bool:
        target = self.song_dir / filename
        return self.play(target.name) if target.is_file() else False
