import importlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TAC3D_SDK_PATH = (
    "/home/mel/ybzhou/lerobot-mujoco-tutorial/local_tac3d_core/"
    "Tac3D-API/python/PyTac3D"
)
TAC3D_SDK_RELATIVE_PATH = (
    "Tac3D-SDK-v3.3.0-20250407/"
    "Tac3D-SDK-v3.3.0/Tac3D-API/python/PyTac3D"
)


@dataclass
class Tac3DFrame:
    displacement: np.ndarray
    distributed_force: np.ndarray
    wrench: np.ndarray
    timestamp: float
    valid: bool


class Tac3DFrameCache:
    """Receive Tac3D Desktop UDP frames and cache the latest frame by SN."""

    def __init__(
        self,
        sdk_path: str = DEFAULT_TAC3D_SDK_PATH,
        module_name: str = "PyTac3D",
        udp_port: int = 9988,
        max_queue_size: int = 5,
    ) -> None:
        self._lock = threading.Lock()
        self._frames_by_sn: dict[str, dict[str, Any]] = {}
        self._sn_order: list[str] = []
        self._sensor = self._build_sensor(
            sdk_path=sdk_path,
            module_name=module_name,
            udp_port=udp_port,
            max_queue_size=max_queue_size,
        )

    def get_latest(self, device_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            if not self._frames_by_sn:
                return None

            sn = self._resolve_sn(device_id)
            if sn is None:
                return None
            frame = self._frames_by_sn.get(sn)
            if frame is None:
                return None
            return dict(frame)

    def wait_for_any_frame(self, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frames_by_sn:
                    return True
            time.sleep(0.02)
        return False

    def available_sensor_ids(self) -> list[str]:
        with self._lock:
            return list(self._sn_order)

    def close(self) -> None:
        close = getattr(self._sensor, "close", None)
        if callable(close):
            close()

    def _build_sensor(
        self,
        sdk_path: str,
        module_name: str,
        udp_port: int,
        max_queue_size: int,
    ) -> Any:
        path, module_name = self._resolve_sdk_path(sdk_path, module_name)
        if path is not None:
            sys.path.insert(0, str(path.parent if path.is_file() else path))

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            if missing == "ruamel":
                install_hint = "pip install ruamel.yaml"
            elif missing == "cv2":
                install_hint = "pip install opencv-python"
            elif missing == module_name:
                searched = ", ".join(str(path) for path in self._candidate_sdk_paths(sdk_path))
                install_hint = (
                    "mount the Tac3D SDK drive or pass "
                    "--tactile-sdk-path /path/to/Tac3D-API/python/PyTac3D"
                )
                raise ModuleNotFoundError(
                    f"Tac3D SDK module {module_name!r} was not found. "
                    f"Searched SDK paths: {searched}. {install_hint}."
                ) from exc
            else:
                install_hint = f"pip install {missing}"
            raise ModuleNotFoundError(
                f"Tac3D SDK import failed because Python module {missing!r} "
                f"is missing. Install it in the same environment: {install_hint}"
            ) from exc
        sensor_cls = getattr(module, "Sensor")
        try:
            return sensor_cls(
                recvCallback=self._recv_callback,
                port=udp_port,
                maxQSize=max_queue_size,
            )
        except OSError as exc:
            raise OSError(
                f"Failed to bind Tac3D UDP receiver on port {udp_port}. "
                "Close PyTac3D_Displayer2.py or any other process using that port, "
                "or start Tac3D/launch_nodes with a different UDP port."
            ) from exc

    def _resolve_sdk_path(
        self,
        sdk_path: str,
        module_name: str,
    ) -> tuple[Path | None, str]:
        for path in self._candidate_sdk_paths(sdk_path):
            path = path.expanduser()
            if path.is_file():
                return path, path.stem
            if path.is_dir():
                return path, module_name
        return None, module_name

    def _candidate_sdk_paths(self, sdk_path: str) -> list[Path]:
        candidates = []
        if sdk_path:
            candidates.append(Path(sdk_path))

        repo_local = Path(__file__).resolve().parents[2] / "local_tac3d_py"
        candidates.append(repo_local)

        media_root = Path("/media/mel")
        if media_root.exists():
            candidates.extend(media_root.glob(f"*/{TAC3D_SDK_RELATIVE_PATH}"))

        deduped = []
        seen = set()
        for path in candidates:
            key = str(path)
            if key not in seen:
                deduped.append(path)
                seen.add(key)
        return deduped

    def _recv_callback(self, frame: dict[str, Any], _param: Any) -> None:
        sn = frame.get("SN")
        if not sn:
            return

        with self._lock:
            if sn not in self._frames_by_sn:
                self._sn_order.append(sn)
                print(f"Tac3D sensor connected: {sn}")
            self._frames_by_sn[sn] = frame

    def _resolve_sn(self, device_id: str) -> str | None:
        if not device_id:
            return self._sn_order[0] if self._sn_order else None
        if device_id in self._frames_by_sn:
            return device_id
        if device_id.isdigit():
            index = int(device_id)
            if 0 <= index < len(self._sn_order):
                return self._sn_order[index]
        return None


class Tac3DDriver:
    """Per-sensor view over a shared Tac3DFrameCache."""

    def __init__(
        self,
        frame_cache: Tac3DFrameCache,
        device_id: str = "",
        max_points: int = 400,
        read_timeout_s: float = 10.0,
    ) -> None:
        self.device_id = device_id
        self.max_points = max_points
        self.read_timeout_s = read_timeout_s
        self._frame_cache = frame_cache

    def read(self) -> dict[str, np.ndarray | float | bool]:
        raw_frame = self._wait_for_latest()
        if raw_frame is None:
            available = self._frame_cache.available_sensor_ids()
            available_text = ", ".join(available) if available else "none"
            raise RuntimeError(
                "No Tac3D frame available for device "
                f"{self.device_id!r}. Use a real SN or an index like 0/1, "
                "and make sure Tac3D Desktop is streaming to the configured UDP port. "
                f"Waited {self.read_timeout_s:.1f}s. "
                f"Currently received Tac3D SNs: {available_text}."
            )
        frame = self._normalize_frame(raw_frame)
        return {
            "displacement": frame.displacement,
            "distributed_force": frame.distributed_force,
            "wrench": frame.wrench,
            "timestamp": frame.timestamp,
            "valid": frame.valid,
        }

    def close(self) -> None:
        pass

    def _wait_for_latest(self) -> dict[str, Any] | None:
        deadline = time.time() + self.read_timeout_s
        while True:
            raw_frame = self._frame_cache.get_latest(self.device_id)
            if raw_frame is not None:
                return raw_frame
            if time.time() >= deadline:
                return None
            time.sleep(0.02)

    def _normalize_frame(self, raw_frame: dict[str, Any]) -> Tac3DFrame:
        displacement = self._fixed_points(raw_frame.get("3D_Displacements"))
        distributed_force = self._fixed_points(raw_frame.get("3D_Forces"))

        resultant_force = self._vector3(raw_frame.get("3D_ResultantForce"))
        if not np.any(resultant_force):
            resultant_force = distributed_force.sum(axis=0)
        resultant_moment = self._vector3(raw_frame.get("3D_ResultantMoment"))
        wrench = np.concatenate([resultant_force, resultant_moment]).astype(np.float32)

        timestamp = raw_frame.get("sendTimestamp", raw_frame.get("recvTimestamp", time.time()))
        return Tac3DFrame(
            displacement=displacement,
            distributed_force=distributed_force,
            wrench=wrench,
            timestamp=float(timestamp),
            valid=True,
        )

    def _fixed_points(self, value: Any) -> np.ndarray:
        if value is None:
            raise KeyError("Tac3D frame is missing required 3D field")

        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            if array.size % 3 != 0:
                raise ValueError(f"Expected 3D point vector, got shape {array.shape}")
            array = array.reshape(-1, 3)
        elif array.shape[-1] == 3:
            array = array.reshape(-1, 3)
        elif array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1).reshape(-1, 3)
        else:
            raise ValueError(f"Expected Tac3D field with 3 components, got shape {array.shape}")

        fixed = np.zeros((self.max_points, 3), dtype=np.float32)
        count = min(array.shape[0], self.max_points)
        fixed[:count] = array[:count]
        return fixed

    def _vector3(self, value: Any) -> np.ndarray:
        if value is None:
            return np.zeros(3, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size < 3:
            array = np.pad(array, (0, 3 - array.size))
        return array[:3]
