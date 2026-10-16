"""Pygame renderer for EMO's blue rounded-rectangle eyes."""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger(__name__)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


class PygameFace:
    """Port of the supplied 320x240 eye animation. Pygame is imported lazily."""

    def __init__(self, status=None, width: int = 320, height: int = 240,
                 fullscreen: bool = True, driver: str = "auto", on_action=None) -> None:
        self.status = status
        self.width = int(width)
        self.height = int(height)
        self.fullscreen = bool(fullscreen)
        self.driver = (driver or "auto").strip().lower()
        self.on_action = on_action
        self.surface = None
        self._pygame = None
        self._clock = None
        self._lock = threading.RLock()
        self.state = "idle"
        self.state_started = time.monotonic()
        self.state_until = None
        self.talking = False
        self.is_talking = False
        self.mood = 0
        self.anim_mode = None
        self._base_w, self._base_h, self._space, self._radius = 90, 90, 14, 22
        self._center_x = (320 - (self._base_w * 2 + self._space)) // 2
        self._center_y = (240 - self._base_h) // 2
        self._left_x = float(self._center_x)
        self._left_y = float(self._center_y)
        self._left_x_next = self._left_x
        self._left_y_next = self._left_y
        self._right_x = float(self._center_x + self._base_w + self._space)
        self._right_y = float(self._center_y)
        self._right_x_next = self._right_x
        self._right_y_next = self._right_y
        self._left_h = float(self._base_h)
        self._right_h = float(self._base_h)
        self._last_frame_ms = time.monotonic() * 1000
        self._next_blink_ms = self._last_frame_ms + 2000
        self._next_idle_ms = self._last_frame_ms + 3000
        self._closed = False

    def start(self) -> None:
        if self.surface is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Pygame is not installed. Run bash setup_emo_robot.sh first.") from exc
        # SPI LCDs need a Linux framebuffer/kernel driver configured separately.
        if self.driver in {"fbcon", "framebuffer"}:
            os.environ.setdefault("SDL_VIDEODRIVER", "fbcon")
            os.environ.setdefault("SDL_FBDEV", "/dev/fb1")
            os.environ.setdefault("SDL_NOMOUSE", "1")
        elif self.driver == "kmsdrm":
            os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")
        self._pygame = pygame
        pygame.init()
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.surface = pygame.display.set_mode((self.width, self.height), flags)
        self.width, self.height = self.surface.get_size()
        pygame.display.set_caption("EMO")
        pygame.mouse.set_visible(not self.fullscreen)
        self._clock = pygame.time.Clock()
        self._closed = False

    def set_state(self, state: str, duration: float | None = None) -> None:
        with self._lock:
            self.state = str(state)
            self.state_started = time.monotonic()
            self.state_until = (
                self.state_started + max(0.0, float(duration))
                if duration is not None else None
            )

    def get_state(self) -> str:
        with self._lock:
            if self.state_until is not None and time.monotonic() >= self.state_until:
                self.state = "idle"
                self.state_until = None
            return self.state

    def set_talking(self, active: bool) -> None:
        with self._lock:
            self.talking = bool(active)
            self.is_talking = bool(active)

    def trigger_anim(self, anim_name: str) -> None:
        self.anim_mode = anim_name
        self.set_state(anim_name)

    def setMood(self, mood: int) -> None:
        self.mood = int(mood)

    def update(self) -> bool:
        if self.surface is None or self._pygame is None:
            return True
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN) and self.on_action:
                    self.on_action()
            elif event.type == pygame.MOUSEBUTTONDOWN and self.on_action:
                self.on_action()

        now = time.monotonic() * 1000
        if now - self._last_frame_ms >= 1000 / 60:
            self._logic(now)
            self._render(now)
            self._last_frame_ms = now
        self._clock.tick(60)
        pygame.display.flip()
        return True

    @staticmethod
    def _rand_int(low: int, high: int) -> int:
        import random
        return random.randint(low, high)

    def _logic(self, now: float) -> None:
        state = self.get_state()
        with self._lock:
            active_anim = self.anim_mode is not None or state not in {"idle", "listening"}
            talking = self.talking
        if now >= self._next_blink_ms and not active_anim and not talking:
            self._left_h = 1
            self._right_h = 1
            self._next_blink_ms = now + self._rand_int(2000, 5000)

        if now >= self._next_idle_ms and not active_anim:
            max_x = 320 - (self._base_w * 2 + self._space)
            max_y = 240 - self._base_h
            self._left_x_next = float(self._rand_int(0, max(1, max_x)))
            self._left_y_next = float(self._rand_int(0, max(1, max_y)))
            self._next_idle_ms = now + self._rand_int(3000, 7000)

        if talking:
            self._left_h = max(10, self._base_h - self._rand_int(10, 40))
            self._right_h = max(10, self._base_h - self._rand_int(10, 40))
        else:
            self._left_h = (self._left_h + self._base_h) / 2
            self._right_h = (self._right_h + self._base_h) / 2

        self._left_x = (self._left_x + self._left_x_next) / 2
        self._left_y = (self._left_y + self._left_y_next) / 2
        self._right_x_next = self._left_x_next + self._base_w + self._space
        self._right_y_next = self._left_y_next
        self._right_x = (self._right_x + self._right_x_next) / 2
        self._right_y = (self._right_y + self._right_y_next) / 2

    def _render(self, now: float) -> None:
        pygame = self._pygame
        surface = self.surface
        surface.fill(BLACK)
        sx, sy = self.width / 320, self.height / 240
        listening = self.get_state() == "listening" or self.anim_mode == "listening"
        y_offset = math_sin(now * 0.01) * 4 if listening else 0
        radius = max(1, int(self._radius * min(sx, sy)))
        left = (
            int(self._left_x * sx),
            int(self._left_y * sy + y_offset * sy),
            max(1, int(self._base_w * sx)),
            max(1, int(self._left_h * sy)),
        )
        right = (
            int(self._right_x * sx),
            int(self._right_y * sy + y_offset * sy),
            max(1, int(self._base_w * sx)),
            max(1, int(self._right_h * sy)),
        )
        pygame.draw.rect(surface, BLUE, left, border_radius=radius)
        pygame.draw.rect(surface, BLUE, right, border_radius=radius)

    def close(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
                self._pygame.quit()
            except Exception:
                log.debug("Pygame shutdown raised", exc_info=True)
        self.surface = None
        self._pygame = None
        self._closed = True


def math_sin(value: float) -> float:
    import math
    return math.sin(value)
