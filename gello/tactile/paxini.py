import threading
import time

import numpy as np


PAXINI_DISTRIBUTED_LENGTHS = {
    "S1813_elite": 93,
    "S2015_elite": 156,
    "S1813_core": 153,
    "S2716_core": 348,
    "S3013_core": 288,
    "M2826_omega": 381,
    "L3530_omega": 405,
    "S1610_elite": 75,
    "M2324_core": 204,
    "M3025_core": 231,
    "L5325_omega": 717,
    "M2020_elite": 27,
}
PAXINI_DEFAULT_MODEL = "S2716_core"
PAXINI_PORT_READY_DELAY = 2.5


def empty_tactile_frame(max_points: int) -> dict[str, np.ndarray | float | bool]:
    return {
        "displacement": np.zeros((max_points, 3), dtype=np.float32),
        "distributed_force": np.zeros((max_points, 3), dtype=np.float32),
        "wrench": np.zeros(6, dtype=np.float32),
        "timestamp": 0.0,
        "valid": False,
    }


class EmptyTactileReader:
    def __init__(self, max_points: int = 400) -> None:
        self._frame = empty_tactile_frame(max_points)

    def read(self) -> dict[str, np.ndarray | float | bool]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._frame.items()
        }

    def close(self) -> None:
        pass


class PaxiniS2716TactileReader:
    """Read Paxini/PX-6AX tactile data over USB serial."""

    def __init__(
        self,
        port: str | None,
        *,
        model: str = PAXINI_DEFAULT_MODEL,
        module_id: str = "02",
        device_addr: str | None = None,
        baudrate: int = 921600,
        timeout_s: float = 0.1,
        read_timeout_s: float = 0.05,
        probe_address: bool = True,
        calibrate_on_start: bool = False,
        ready_delay_s: float = PAXINI_PORT_READY_DELAY,
    ) -> None:
        try:
            import serial
            import serial.tools.list_ports
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading Paxini tactile sensors requires pyserial. Install it with: "
                "pip install pyserial"
            ) from exc

        self._serial_module = serial
        self.model = model
        self.distributed_length = int(PAXINI_DISTRIBUTED_LENGTHS[model])
        self.resultant_length = 3
        self.read_timeout_s = float(read_timeout_s)
        self.device_addr = (device_addr or self._device_addr_from_module(module_id)).upper()
        self.port = port or self._find_serial_port(serial)
        if self.port is None:
            raise RuntimeError(
                "No USB serial port found for Paxini tactile sensor. Pass "
                "--paxini-left-port /dev/ttyUSBX explicitly."
            )

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout_s,
                write_timeout=timeout_s,
                inter_byte_timeout=0.0005,
                xonxoff=False,
                rtscts=False,
            )
        except serial.SerialException as exc:
            raise RuntimeError(f"Failed to open Paxini serial port {self.port}: {exc}") from exc

        time.sleep(float(ready_delay_s))
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        if probe_address:
            self._probe_device_address()
        if calibrate_on_start:
            self.calibrate()
        print(
            "Paxini tactile serial ready: "
            f"port={self.port}, model={self.model}, device_addr={self.device_addr}, "
            f"distributed_length={self.distributed_length}"
        )

    @staticmethod
    def _device_addr_from_module(module_id: str) -> str:
        return f"{int(module_id, 16) + 1:02X}"

    def _find_serial_port(self, serial_module) -> str | None:
        for port_info in serial_module.tools.list_ports.comports():
            port_name = port_info.device
            description = port_info.description or ""
            if "USB" in description or "ttyUSB" in port_name or "ttyACM" in port_name:
                return port_name
        return None

    def _commands(self) -> dict[str, str]:
        length_low = self.distributed_length & 0xFF
        length_high = (self.distributed_length >> 8) & 0xFF
        return {
            "calibration": f"55 AA 0A 00 {self.device_addr} 00 79 03 00 00 00 01 00 01",
            "resultant_force": f"55 AA 09 00 {self.device_addr} 00 FB F0 03 00 00 03 00",
            "distributed_force": (
                f"55 AA 09 00 {self.device_addr} 00 FB 0E 04 00 00 "
                f"{length_low:02X} {length_high:02X}"
            ),
        }

    @staticmethod
    def _calculate_lrc(data: bytes) -> int:
        lrc = 0
        for byte in data:
            lrc = (lrc + byte) & 0xFF
        return ((~lrc) + 1) & 0xFF

    def _build_frame_with_lrc(self, frame: str) -> bytes:
        frame_bytes = bytes.fromhex(frame.replace(" ", ""))
        lrc = self._calculate_lrc(frame_bytes)
        return frame_bytes + bytes([lrc])

    def _read_response(self, timeout: float | None = None) -> bytes | None:
        response = b""
        start_time = time.time()
        timeout = self.read_timeout_s if timeout is None else float(timeout)
        while time.time() - start_time < timeout:
            waiting = self.ser.in_waiting
            if waiting > 0:
                response += self.ser.read(waiting)
                if len(response) >= 4 and response[:2].hex() == "aa55":
                    response_length = int.from_bytes(response[2:4], byteorder="little")
                    expected_total = 4 + response_length + 1
                    if len(response) >= expected_total:
                        return response[:expected_total]
                start_time = time.time()
            time.sleep(0.001)
        return response if response else None

    def _send_command(self, command_type: str, timeout: float | None = None) -> bytes | None:
        commands = self._commands()
        if command_type not in commands:
            raise ValueError(f"Unknown Paxini command type: {command_type}")
        try:
            self.ser.reset_input_buffer()
            self.ser.write(self._build_frame_with_lrc(commands[command_type]))
            time.sleep(0.01)
            return self._read_response(timeout=timeout)
        except self._serial_module.SerialException as exc:
            raise RuntimeError(f"Paxini command {command_type} failed on {self.port}: {exc}") from exc

    def _probe_device_address(self) -> None:
        original_addr = self.device_addr
        for device_addr in range(1, 9):
            self.device_addr = f"{device_addr:02X}"
            response = self._send_command("resultant_force", timeout=0.3)
            if response and response[:2].hex() == "aa55":
                return
        self.device_addr = original_addr

    def calibrate(self) -> None:
        response = self._send_command("calibration", timeout=0.5)
        if not response or response[:2].hex() != "aa55":
            raise RuntimeError(f"Paxini calibration failed on {self.port}")

    @staticmethod
    def _parse_xyz_triplets(data: bytes) -> np.ndarray:
        raw = np.frombuffer(data, dtype=np.uint8)
        triplet_count = raw.size // 3
        raw = raw[: triplet_count * 3].reshape(triplet_count, 3)
        parsed = raw.astype(np.int16)
        signed_xy = parsed[:, :2]
        signed_xy[signed_xy > 127] -= 256
        parsed[:, :2] = signed_xy
        return parsed.astype(np.float32) * 0.1

    def read(self) -> dict[str, np.ndarray | float | bool]:
        resultant = np.zeros(3, dtype=np.float32)
        result_response = self._send_command("resultant_force")
        if result_response and len(result_response) >= 14 + self.resultant_length:
            result_data = result_response[14 : 14 + self.resultant_length]
            resultant = self._parse_xyz_triplets(result_data).reshape(-1)[:3].astype(np.float32)

        distributed = np.zeros((0, 3), dtype=np.float32)
        dist_response = self._send_command("distributed_force")
        if dist_response and len(dist_response) >= 14:
            dist_data = dist_response[14 : 14 + self.distributed_length]
            distributed = self._parse_xyz_triplets(dist_data).astype(np.float32)

        displacement = np.zeros_like(distributed, dtype=np.float32)
        wrench = np.zeros(6, dtype=np.float32)
        wrench[:3] = resultant
        valid = distributed.size > 0 or bool(np.any(resultant))
        return {
            "displacement": displacement,
            "distributed_force": distributed,
            "wrench": wrench,
            "timestamp": time.time() if valid else 0.0,
            "valid": valid,
        }

    def close(self) -> None:
        if getattr(self, "ser", None) is not None and self.ser.is_open:
            self.ser.close()


class AsyncTactileReader:
    def __init__(
        self,
        tactile_reader,
        *,
        max_points: int = 400,
        poll_interval_s: float = 0.0,
        name: str = "tactile",
    ) -> None:
        self._reader = tactile_reader
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._name = name
        self._fallback = empty_tactile_frame(max_points)
        self._latest = None
        self._last_error_print = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _copy_frame(self, frame: dict[str, np.ndarray | float | bool]) -> dict:
        copied = {}
        for key, value in frame.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._reader.read()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error_print > 2.0:
                    print(f"{self._name} async read failed: {exc}")
                    self._last_error_print = now
                time.sleep(0.02)
                continue

            if not bool(frame.get("valid", True)):
                time.sleep(0.02)
                continue

            with self._lock:
                self._latest = self._copy_frame(frame)

            if self._poll_interval_s > 0.0:
                self._stop_event.wait(self._poll_interval_s)

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.01)
        return False

    def read(self) -> dict[str, np.ndarray | float | bool]:
        with self._lock:
            if self._latest is None:
                return self._copy_frame(self._fallback)
            return self._copy_frame(self._latest)

    def close(self) -> None:
        self._stop_event.set()
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()
        self._thread.join(timeout=1.0)
