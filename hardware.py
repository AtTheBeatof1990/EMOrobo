"""BCM GPIO controller for four servo channels, touch input, and Pi thermal monitor."""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class HardwareController:
    """Servo order: head yaw, head pitch, left arm, right arm.

    Servos must have a separate regulated supply with common ground to the Pi.
    The pins here use BCM numbering, not physical-header numbering.
    """
    SERVO_NAMES = ("head_yaw", "head_pitch", "left_arm", "right_arm")
    EXPRESSION_POSES = {
        "idle": (0, 0, 0, 0), "listening": (0, -8, 0, 0),
        "thinking": (12, 8, 0, 0), "happy": (0, -4, -18, 18),
        "excited": (0, -12, 38, -38), "dance": (-15, 0, 38, -38),
        "sad": (0, -20, -12, -12), "angry": (0, 12, 18, 18),
        "surprised": (0, 12, 28, -28), "overheat": (0, 0, 0, 0),
        "talking": (0, 3, 4, -4), "curious": (16, -5, 0, 8),
        "sleepy": (0, -15, -8, -8),
    }

    def __init__(self, touch_gpio: int = 16, servo_gpios=(5, 6, 12, 13),
                 touch_power_gpio: int = 19, mpu_power_gpio: int = 26,
                 status=None, on_touch=None) -> None:
        self.touch_gpio = int(touch_gpio)
        self.touch_power_gpio = int(touch_power_gpio)
        self.mpu_power_gpio = int(mpu_power_gpio)
        self.servo_gpios = tuple(int(pin) for pin in servo_gpios)
        if len(self.servo_gpios) != 4 or len(set(self.servo_gpios)) != 4:
            raise ValueError("SERVO_GPIOS must contain four distinct BCM GPIO numbers.")
        if any(pin < 0 or pin > 27 for pin in self.servo_gpios):
            raise ValueError("Servo GPIO numbers must be in the BCM range 0-27.")
        self.status = status
        self.on_touch = on_touch
        self.button = None
        self.servos = []
        self._touch_pwr = None
        self._mpu_pwr = None
        self._lock = threading.RLock()

    def start(self) -> tuple[bool, str]:
        enabled, errors = [], []
        try:
            from gpiozero import AngularServo, Button, OutputDevice
        except Exception as exc:
            return False, f"gpiozero unavailable; hardware is disabled: {exc}"
        try:
            # 3.3V Power rail for capacitive touch sensor (Pin 35, GPIO 19)
            if self.touch_power_gpio:
                try:
                    self._touch_pwr = OutputDevice(self.touch_power_gpio, active_high=True, initial_value=True)
                    enabled.append(f"touch 3.3V pwr BCM GPIO {self.touch_power_gpio} (Pin 35)")
                except Exception as exc:
                    errors.append(f"touch power GPIO {self.touch_power_gpio} failed: {exc}")

            # 3.3V Power rail for dual MPU6050 + MPU#2 AD0 High (Pin 37, GPIO 26)
            if self.mpu_power_gpio:
                try:
                    self._mpu_pwr = OutputDevice(self.mpu_power_gpio, active_high=True, initial_value=True)
                    enabled.append(f"MPU 3.3V pwr + AD0 BCM GPIO {self.mpu_power_gpio} (Pin 37)")
                except Exception as exc:
                    errors.append(f"MPU power GPIO {self.mpu_power_gpio} failed: {exc}")

            # Capacitive touch signal (Pin 36, GPIO 16)
            if self.touch_gpio:
                self.button = Button(self.touch_gpio, pull_up=False, bounce_time=0.12)
                self.button.when_pressed = self._touch_pressed
                enabled.append(f"touch signal BCM GPIO {self.touch_gpio} (Pin 36)")

            # Four servo channels (Pins 29, 31, 32, 33)
            for name, pin in zip(self.SERVO_NAMES, self.servo_gpios):
                servo = AngularServo(pin, min_angle=-60, max_angle=60,
                    min_pulse_width=0.0005, max_pulse_width=0.0025, initial_angle=0)
                self.servos.append(servo)
                enabled.append(f"{name} BCM GPIO {pin}")
        except Exception as exc:
            errors.append(str(exc))
            log.warning("GPIO setup partially failed: %s", exc)
            for servo in self.servos:
                try: servo.close()
                except Exception: pass
            self.servos.clear()
        if errors:
            return False, "; ".join(errors)
        return True, "Enabled " + ", ".join(enabled)

    def _touch_pressed(self) -> None:
        if self.status is not None:
            with self.status.lock:
                self.status.last_touch = time.monotonic()
        if self.on_touch is not None:
            try: self.on_touch()
            except Exception: log.exception("Touch callback failed")

    def set_pose(self, pose) -> None:
        values = tuple(max(-60.0, min(60.0, float(value))) for value in pose)
        if len(values) != 4:
            raise ValueError("A pose must contain four angles (yaw, pitch, left arm, right arm).")
        with self._lock:
            for servo, value in zip(self.servos, values):
                try: servo.angle = value
                except Exception as exc: log.debug("Servo movement failed: %s", exc)

    def set_expression(self, expression: str) -> None:
        self.set_pose(self.EXPRESSION_POSES.get(expression, self.EXPRESSION_POSES["idle"]))

    def close(self) -> None:
        with self._lock:
            for device in [*self.servos, self.button, self._touch_pwr, self._mpu_pwr]:
                if device is not None:
                    try: device.close()
                    except Exception: pass
            self.servos.clear()
            self.button = None
            self._touch_pwr = None
            self._mpu_pwr = None


class ThermalMonitor:
    def __init__(self, status, warning_c: float = 75.0, lockout_c: float = 82.0,
                 resume_c: float = 72.0, interval: float = 4.0) -> None:
        self.status, self.warning_c, self.lockout_c = status, warning_c, lockout_c
        self.resume_c, self.interval = resume_c, interval
        self.stop_event = threading.Event()
        self.thread = None
        self._locked_out = False

    @staticmethod
    def read_temperature() -> float:
        for path in (Path("/sys/class/thermal/thermal_zone0/temp"),
                     Path("/sys/devices/virtual/thermal/thermal_zone0/temp")):
            try:
                value = float(path.read_text(encoding="ascii").strip())
                return value / 1000.0 if value > 200 else value
            except Exception:
                pass
        return 0.0

    def start(self) -> None:
        if self.thread and self.thread.is_alive(): return
        self.thread = threading.Thread(target=self._run, daemon=True, name="EMO-Thermal")
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            temperature = self.read_temperature()
            if temperature >= self.lockout_c: self._locked_out = True
            elif self._locked_out and temperature <= self.resume_c: self._locked_out = False
            self.status.set_thermal(temperature, temperature >= self.warning_c, self._locked_out)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
