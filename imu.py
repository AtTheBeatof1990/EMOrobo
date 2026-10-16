"""Two MPU6050 sensors on the Pi's I2C bus (addresses 0x68 and 0x69)."""
from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


def _signed(raw: int) -> int:
    return raw - 65536 if raw > 32767 else raw


class MPU6050Array:
    """Read acceleration and gyro values from multiple MPU6050 devices."""

    PWR_MGMT_1 = 0x6B
    DATA_START = 0x3B

    def __init__(self, addresses=(0x68, 0x69), status=None, interval: float = 0.10,
                 bus_id: int | str = "auto", sda_gpio: int = 20, scl_gpio: int = 21) -> None:
        self.addresses = tuple(int(address) for address in addresses)
        self.status = status
        self.interval = max(0.05, float(interval))
        self.bus_id = str(bus_id).strip().lower() if bus_id is not None else "auto"
        self.sda_gpio = int(sda_gpio)
        self.scl_gpio = int(scl_gpio)
        self.active_bus: int | None = None
        self._bus = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._latest: dict[str, Any] = {}
        self._enabled_addresses: list[int] = []

    def _open_bus(self):
        from smbus2 import SMBus
        if self.bus_id not in {"auto", ""}:
            try:
                target_bus = int(self.bus_id)
                return SMBus(target_bus), target_bus
            except Exception as exc:
                log.warning("Failed to open configured I2C bus %s: %s", self.bus_id, exc)

        # Probing order: Software I2C (bus 3 on GPIO 20/21 via i2c-gpio) first, then hardware I2C (bus 1)
        candidates = [3, 1, 0, 4]
        for candidate in candidates:
            try:
                bus = SMBus(candidate)
                # Test write to any configured address
                for address in self.addresses:
                    try:
                        bus.write_byte_data(address, self.PWR_MGMT_1, 0x00)
                        log.info("MPU6050 found on I2C bus %d at 0x%02X", candidate, address)
                        return bus, candidate
                    except OSError:
                        pass
                bus.close()
            except Exception:
                pass

        # Fallback: attempt bus 3 (software I2C), then bus 1
        for fallback_bus in [3, 1]:
            try:
                return SMBus(fallback_bus), fallback_bus
            except Exception:
                pass
        return SMBus(1), 1

    def start(self) -> tuple[bool, str]:
        try:
            from smbus2 import SMBus
            self._bus, self.active_bus = self._open_bus()
            for address in self.addresses:
                try:
                    self._bus.write_byte_data(address, self.PWR_MGMT_1, 0x00)
                    self._enabled_addresses.append(address)
                except OSError as exc:
                    log.warning("MPU6050 at 0x%02X is unavailable on I2C bus %s: %s", address, self.active_bus, exc)
            if not self._enabled_addresses:
                self.stop()
                return False, (
                    f"No MPU6050 found on I2C bus {self.active_bus}. Check Software I2C wiring: "
                    f"SDA=GPIO {self.sda_gpio} (Pin 38), SCL=GPIO {self.scl_gpio} (Pin 40), "
                    f"Power=GPIO 26 (Pin 37), and AD0 for 0x68/0x69."
                )
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="EMO-MPU6050")
            self._thread.start()
            addrs = ", ".join(f"0x{address:02X}" for address in self._enabled_addresses)
            return True, f"MPU6050 active at {addrs} on I2C bus {self.active_bus} (Software I2C SDA={self.sda_gpio}, SCL={self.scl_gpio})"
        except Exception as exc:
            self.stop()
            return False, f"MPU6050 unavailable: {exc}"

    def read_sensor(self, address: int) -> dict[str, float]:
        if self._bus is None:
            raise RuntimeError("I2C bus is not open")
        data = self._bus.read_i2c_block_data(address, self.DATA_START, 14)
        values = [_signed((data[index] << 8) | data[index + 1]) for index in range(0, 14, 2)]
        ax, ay, az, _temperature, gx, gy, gz = values
        return {
            "accel_x_g": ax / 16384.0,
            "accel_y_g": ay / 16384.0,
            "accel_z_g": az / 16384.0,
            "gyro_x_dps": gx / 131.0,
            "gyro_y_dps": gy / 131.0,
            "gyro_z_dps": gz / 131.0,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {}
            for address in self._enabled_addresses:
                try:
                    sample[f"0x{address:02X}"] = self.read_sensor(address)
                except Exception as exc:
                    log.debug("MPU6050 read failed for 0x%02X: %s", address, exc)
            with self._lock:
                self._latest = sample
            if self.status is not None:
                self.status.set_imu(sample)
            self._stop.wait(self.interval)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        self._enabled_addresses.clear()
