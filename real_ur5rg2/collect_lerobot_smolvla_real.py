import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.zmq_core.camera_node import ZMQClientCamera
from gello.force.robotiq_fts300 import (
    RobotiqFTS300ModbusSerial,
    RobotiqFTS300ModbusTCP,
)
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.tactile_node import ZMQClientTactile


TACTILE_FIELD_KEYS = (
    "observation.tactile_left.displacement",
    "observation.tactile_left.distributed_force",
    "observation.tactile_right.displacement",
    "observation.tactile_right.distributed_force",
)
TACTILE_DATASET_KEYS = TACTILE_FIELD_KEYS + (
    "observation.tactile_left.wrench",
    "observation.tactile_right.wrench",
    "observation.tactile.timestamp",
)
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
FORCE_TORQUE_KEY = "observation.force_torque"
FORCE_TARE_SECONDS = 2.0
DEFAULT_FORCE_SAFETY_THRESHOLD_N = 5.0
DEFAULT_FORCE_SAFETY_HARD_STOP_N = 18.0
DEFAULT_TORQUE_SAFETY_THRESHOLD_NM = 0.5
DEFAULT_TORQUE_SAFETY_HARD_STOP_NM = 1.5
DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M = 0.042 - 0.0034
RESET_TCP_POSITION = np.asarray([-0.040076405, -0.5464264, 0.21074047], dtype=np.float64)
RESET_TCP_POSITION_RANDOM_RANGE_M = 0.1
# The bolt/tool axis on this TCP points along local -Z, so identity makes it
# point vertically down in the base frame.
RESET_TCP_ROTATION_VECTOR = np.asarray(
    [-np.pi, 0.0, 0.0],
    dtype=np.float64,
)

_DEFAULT_COMPUTE_EPISODE_STATS = lerobot_dataset_module.compute_episode_stats
_DEFAULT_AGGREGATE_STATS = lerobot_dataset_module.aggregate_stats


def _channelwise_tactile_stats(feature_stats: dict) -> dict:
    if not feature_stats:
        return feature_stats

    mean = np.asarray(feature_stats.get("mean"))
    if mean.ndim != 2 or mean.shape[-1] != 3:
        return feature_stats

    channel_mean = np.mean(mean, axis=0)
    channel_std = None
    if "std" in feature_stats:
        std = np.asarray(feature_stats["std"])
        if std.shape == mean.shape:
            variance = np.mean(std**2 + (mean - channel_mean) ** 2, axis=0)
            channel_std = np.sqrt(variance)

    reduced_stats = dict(feature_stats)
    if "min" in feature_stats:
        reduced_stats["min"] = np.min(np.asarray(feature_stats["min"]), axis=0)
    if "max" in feature_stats:
        reduced_stats["max"] = np.max(np.asarray(feature_stats["max"]), axis=0)
    reduced_stats["mean"] = channel_mean
    if channel_std is not None:
        reduced_stats["std"] = channel_std
    return reduced_stats


def aggregate_stats_with_channelwise_tactile(
    stats_list: list[dict[str, dict]],
) -> dict[str, dict[str, np.ndarray]]:
    normalized_stats = []
    for stats in stats_list:
        normalized = dict(stats)
        for key in TACTILE_FIELD_KEYS:
            if key in normalized:
                normalized[key] = _channelwise_tactile_stats(normalized[key])
        normalized_stats.append(normalized)

    return _DEFAULT_AGGREGATE_STATS(normalized_stats)


def compute_episode_stats_with_channelwise_tactile(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
) -> dict:
    ep_stats = _DEFAULT_COMPUTE_EPISODE_STATS(episode_data, features)

    for key in TACTILE_FIELD_KEYS:
        if key not in episode_data:
            continue

        tactile_array = np.asarray(episode_data[key], dtype=np.float32)
        if tactile_array.ndim != 3 or tactile_array.shape[-1] != 3:
            continue

        ep_stats[key] = {
            "min": np.min(tactile_array, axis=(0, 1)),
            "max": np.max(tactile_array, axis=(0, 1)),
            "mean": np.mean(tactile_array, axis=(0, 1)),
            "std": np.std(tactile_array, axis=(0, 1)),
            "count": np.array([len(tactile_array)]),
        }

    return ep_stats


lerobot_dataset_module.compute_episode_stats = compute_episode_stats_with_channelwise_tactile
lerobot_dataset_module.aggregate_stats = aggregate_stats_with_channelwise_tactile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LeRobot/SmolVLA episodes on a real UR5 + RG2 with keyboard teleop."
    )
    parser.add_argument("--repo-id", default="ur5_rg2_real_smolvla")
    parser.add_argument("--root", default="./real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627_test")
    parser.add_argument("--task", default="Insert the bolt into the nut.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument(
        "--robot-timeout-ms",
        type=int,
        default=10000,
        help="ZMQ timeout for robot requests; first RG2 command can take several seconds.",
    )
    parser.add_argument("--wrist-camera-port", type=int, default=5000)
    parser.add_argument("--base-camera-port", type=int, default=5001)
    parser.add_argument(
        "--collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect tactile fields. Use --no-collect-tactile to disable.",
    )
    parser.add_argument(
        "--tactile-source",
        choices=("tac3d", "paxini"),
        default="tac3d",
        help="Tactile sensor backend: Tac3D ZMQ nodes or Paxini S2716_core serial sensors.",
    )
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-max-points", type=int, default=400)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument("--paxini-left-port", default=None)
    parser.add_argument("--paxini-right-port", default=None)
    parser.add_argument(
        "--paxini-model",
        choices=tuple(PAXINI_DISTRIBUTED_LENGTHS.keys()),
        default=PAXINI_DEFAULT_MODEL,
    )
    parser.add_argument("--paxini-left-module-id", default="02")
    parser.add_argument("--paxini-right-module-id", default="03")
    parser.add_argument("--paxini-left-device-addr", default=None)
    parser.add_argument("--paxini-right-device-addr", default=None)
    parser.add_argument("--paxini-baudrate", type=int, default=921600)
    parser.add_argument("--paxini-timeout-s", type=float, default=0.1)
    parser.add_argument("--paxini-read-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--paxini-probe-address",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan Paxini device addresses 01-08 after opening the serial port.",
    )
    parser.add_argument(
        "--paxini-calibrate-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send Paxini calibration command during warmup.",
    )
    parser.add_argument(
        "--collect-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect force/torque wrench as observation.force_torque.",
    )
    parser.add_argument(
        "--force-source",
        choices=("modbus-serial", "modbus-tcp", "ur-rtde"),
        default="modbus-serial",
        help="Read force from Robotiq FTS-300-S Modbus RTU/TCP or UR RTDE TCP force.",
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-slave-address", type=int, default=9)
    parser.add_argument("--force-serial-register-address", type=int, default=180)
    parser.add_argument("--force-serial-function-code", type=int, default=3)
    parser.add_argument("--force-serial-baudrate", type=int, default=19200)
    parser.add_argument("--force-serial-timeout-s", type=float, default=0.02)
    parser.add_argument("--force-serial-retries", type=int, default=3)
    parser.add_argument(
        "--force-serial-wakeup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send 50 bytes of 0xff before opening the Modbus RTU instrument.",
    )
    parser.add_argument("--force-modbus-host", default="192.168.1.1")
    parser.add_argument("--force-modbus-port", type=int, default=502)
    parser.add_argument("--force-modbus-unit-id", type=int, default=9)
    parser.add_argument("--force-modbus-address", type=int, default=0)
    parser.add_argument(
        "--force-modbus-function",
        choices=("holding", "input"),
        default="input",
    )
    parser.add_argument(
        "--force-modbus-format",
        choices=("int16", "int32", "float32"),
        default="int32",
    )
    parser.add_argument(
        "--force-modbus-word-order",
        choices=("big", "little"),
        default="big",
    )
    parser.add_argument("--force-modbus-scale", type=float, default=1.0)
    parser.add_argument("--force-modbus-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--force-print-every",
        type=int,
        default=1,
        help="Print force/torque every N collected frames when force collection is enabled.",
    )
    parser.add_argument(
        "--force-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When force collection is enabled, apply a task-agnostic Cartesian "
            "admittance correction to reduce excessive x/y/z force and torque."
        ),
    )
    parser.add_argument(
        "--force-safety-threshold-n",
        default=f"{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N}",
        help="Per-axis x,y,z force threshold in N before correction starts.",
    )
    parser.add_argument(
        "--force-safety-hard-stop-n",
        default=f"{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N}",
        help="Per-axis x,y,z hard force limit in N. Above this, teleop motion is suppressed and only retreat is commanded.",
    )
    parser.add_argument(
        "--torque-safety-threshold-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM}",
        help="Per-axis rx,ry,rz torque threshold in Nm before posture correction starts.",
    )
    parser.add_argument(
        "--torque-safety-hard-stop-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM}",
        help="Per-axis rx,ry,rz hard torque limit in Nm. Above this, teleop motion is suppressed and only retreat is commanded.",
    )
    parser.add_argument(
        "--force-torque-tool-offset-m",
        type=float,
        default=DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M,
        help=(
            "Lever arm used by the fts300-tool torque mapping, matching "
            "ur5moverealrobot3.py's d = 0.042 - 0.0034."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-signs",
        default=None,
        help=(
            "Manual per-axis sign correction from the force sensor frame to the "
            "selected force safety frame. Use -1 to flip an axis if the retreat "
            "direction is wrong. By default this is selected from --force-safety-axis-preset."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-order",
        default=None,
        help=(
            "Manual axis reorder from raw force sensor xyz to force safety xyz, "
            "e.g. 0,2,1 maps [Fx,Fy,Fz] to [Fx,Fz,Fy]. By default this is "
            "selected from --force-safety-axis-preset."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-preset",
        choices=("auto", "identity", "fts300-tool"),
        default="auto",
        help=(
            "Force axis mapping preset. auto uses fts300-tool for Modbus FTS-300 "
            "and identity for UR RTDE. fts300-tool follows ur5moverealrobot3.py: "
            "[Fx,Fy,Fz] -> [-Fx,-Fz,-Fy]."
        ),
    )
    parser.add_argument(
        "--force-safety-frame",
        choices=("auto", "tool", "base"),
        default="auto",
        help=(
            "Frame of the signed x/y/z force used by force safety. auto uses "
            "tool for Modbus FTS-300 and base for UR RTDE."
        ),
    )
    parser.add_argument(
        "--force-safety-inv-mass",
        type=float,
        default=0.40,
        help="Inverse mass term for the translational admittance correction.",
    )
    parser.add_argument(
        "--force-safety-damping",
        type=float,
        default=18.0,
        help="Damping term for the translational admittance correction.",
    )
    parser.add_argument(
        "--force-safety-stiffness",
        type=float,
        default=80.0,
        help="Stiffness term pulling the correction back to zero.",
    )
    parser.add_argument(
        "--torque-safety-inv-inertia",
        type=float,
        default=0.08,
        help="Inverse inertia term for the rotational admittance correction.",
    )
    parser.add_argument(
        "--torque-safety-damping",
        type=float,
        default=10.0,
        help="Damping term for the rotational admittance correction.",
    )
    parser.add_argument(
        "--torque-safety-stiffness",
        type=float,
        default=35.0,
        help="Stiffness term pulling the posture correction back to zero.",
    )
    parser.add_argument(
        "--force-safety-max-vel",
        type=float,
        default=0.025,
        help="Max Cartesian retreat velocity in m/s.",
    )
    parser.add_argument(
        "--force-safety-max-correction-m",
        type=float,
        default=0.002,
        help="Max absolute Cartesian correction in m.",
    )
    parser.add_argument(
        "--torque-safety-max-angular-vel",
        type=float,
        default=0.10,
        help="Max rotational retreat velocity in rad/s.",
    )
    parser.add_argument(
        "--torque-safety-max-correction-rad",
        type=float,
        default=0.035,
        help="Max absolute posture correction in rad.",
    )
    parser.add_argument(
        "--force-safety-deadband-n",
        type=float,
        default=0.30,
        help="Force deadband in N to avoid chattering near the threshold.",
    )
    parser.add_argument(
        "--torque-safety-deadband-nm",
        type=float,
        default=0.03,
        help="Torque deadband in Nm to avoid chattering near the threshold.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument("--move-step", type=float, default=0.0005)
    parser.add_argument("--rot-step", type=float, default=0.03)
    parser.add_argument("--shift-speed-scale", type=float, default=5)
    parser.add_argument("--screw-pitch", type=float, default=0.0025)
    parser.add_argument(
        "--screw-rot-step",
        type=float,
        default=0.07,
        help="Screw rotation increment in rad per control frame when --screw-rpm is unset.",
    )
    parser.add_argument(
        "--screw-rpm",
        type=float,
        default=None,
        help="Screw speed in revolutions per minute; overrides --screw-rot-step.",
    )
    parser.add_argument(
        "--reset-max-joint-step-rad",
        type=float,
        default=0.3,
        help="Max joint step per control frame when moving to L reset through IK.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-command", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--no-tactile",
        action="store_false",
        dest="collect_tactile",
        help="Alias for --no-collect-tactile.",
    )
    return parser.parse_args()


def make_dataset(args: argparse.Namespace) -> LeRobotDataset:
    root = Path(args.root)
    complete_dataset = (root / "meta").exists() and (root / "data").exists()

    if args.overwrite and root.exists():
        shutil.rmtree(root)
        complete_dataset = False

    if complete_dataset and args.resume:
        print(f"Load existing dataset from {root}")
        return LeRobotDataset(args.repo_id, root=str(root))

    if root.exists():
        if complete_dataset:
            print(f"Directory {root} already contains a LeRobot dataset.")
            ans = input("Overwrite it, continue saving, or quit? [o/c/q] ").strip().lower()
            if ans in ("c", "continue", "r", "resume"):
                print(f"Load existing dataset from {root}")
                return LeRobotDataset(args.repo_id, root=str(root))
            if ans in ("o", "overwrite", "y", "yes"):
                shutil.rmtree(root)
            else:
                raise SystemExit("Quit without changing the dataset.")
        else:
            print(f"Directory {root} exists but is not a complete LeRobot dataset.")
            ans = input("Overwrite it and create a new dataset, or quit? [o/q] ").strip().lower()
            if ans in ("o", "overwrite", "y", "yes"):
                shutil.rmtree(root)
            else:
                raise SystemExit("Quit without changing the dataset.")

    print(f"Create new dataset at {root}")
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "obj_init": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["obj_init"],
        },
    }

    if args.collect_tactile:
        features.update(
            {
                "observation.tactile_left.displacement": {
                    "dtype": "float32",
                    "shape": (args.tactile_max_points, 3),
                    "names": ["taxel", "axis"],
                },
                "observation.tactile_left.distributed_force": {
                    "dtype": "float32",
                    "shape": (args.tactile_max_points, 3),
                    "names": ["taxel", "axis"],
                },
                "observation.tactile_left.wrench": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_torque"],
                },
                "observation.tactile_right.displacement": {
                    "dtype": "float32",
                    "shape": (args.tactile_max_points, 3),
                    "names": ["taxel", "axis"],
                },
                "observation.tactile_right.distributed_force": {
                    "dtype": "float32",
                    "shape": (args.tactile_max_points, 3),
                    "names": ["taxel", "axis"],
                },
                "observation.tactile_right.wrench": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_torque"],
                },
                "observation.tactile.timestamp": {
                    "dtype": "float64",
                    "shape": (2,),
                    "names": ["sensor"],
                },
            }
        )

    if args.collect_force:
        features["observation.force_torque"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["force_torque"],
        }

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        robot_type="ur5_rg2",
        fps=args.fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize((image_size, image_size))
    return np.asarray(image, dtype=np.uint8)


def reset_episode_buffer(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset.clear_episode_buffer()


def dataset_has_tactile_features(dataset: LeRobotDataset) -> bool:
    return any(key in dataset.features for key in TACTILE_DATASET_KEYS)


def dataset_has_force_feature(dataset: LeRobotDataset) -> bool:
    return FORCE_TORQUE_KEY in dataset.features


def validate_requested_sensor_features(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    store_tactile = dataset_has_tactile_features(dataset)
    store_force = dataset_has_force_feature(dataset)

    if args.collect_tactile and not store_tactile:
        raise RuntimeError(
            "This dataset was created without tactile features, but "
            "--collect-tactile is enabled. Use --overwrite, a new --root, or "
            "run with --no-collect-tactile."
        )
    if args.collect_force and not store_force:
        raise RuntimeError(
            "This dataset was created without observation.force_torque, but "
            "--collect-force is enabled. Use --overwrite, a new --root, or "
            "run with --no-collect-force."
        )

    return store_tactile, store_force


def format_force_torque(force_torque: np.ndarray) -> np.ndarray:
    array = np.asarray(force_torque, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


def print_force_torque(force_torque: np.ndarray) -> None:
    values = [float(v) for v in force_torque]
    print(
        "FTS-300-S force_torque "
        f"[Fx,Fy,Fz,Tx,Ty,Tz]=[{values[0]: .2f}, {values[1]: .2f}, "
        f"{values[2]: .2f}, {values[3]: .2f}, {values[4]: .2f}, {values[5]: .2f}]"
    )


def parse_xyz_vector(value: str, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        parts = [float(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be three comma-separated numbers, got {value!r}") from exc
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise ValueError(f"{name} must contain one value or three comma-separated values.")
    array = np.asarray(parts, dtype=np.float64)
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive.")
    return array


def parse_axis_order(value: str) -> np.ndarray:
    try:
        order = [int(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--force-safety-axis-order must be three comma-separated integers, got {value!r}"
        ) from exc
    if sorted(order) != [0, 1, 2]:
        raise ValueError("--force-safety-axis-order must be a permutation of 0,1,2.")
    return np.asarray(order, dtype=int)


def resolve_force_safety_axis_mapping(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    preset = args.force_safety_axis_preset
    if preset == "auto":
        if args.force_source in ("modbus-serial", "modbus-tcp"):
            preset = "fts300-tool"
        else:
            preset = "identity"

    if preset == "fts300-tool":
        # Matches getPureForceTorqueToTool() in ur5moverealrobot3.py:
        # raw [Fx,Fy,Fz] -> tool [-Fx,-Fz,-Fy].
        axis_order = np.asarray([0, 1, 2], dtype=int)
        axis_signs = np.asarray([1.0, 1.0, -1.0], dtype=np.float64)
    elif preset == "identity":
        axis_order = np.asarray([0, 1, 2], dtype=int)
        axis_signs = np.ones(3, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported force safety axis preset: {preset}")

    if args.force_safety_axis_order is not None:
        axis_order = parse_axis_order(args.force_safety_axis_order)
    if args.force_safety_axis_signs is not None:
        axis_signs = parse_xyz_vector(args.force_safety_axis_signs, "--force-safety-axis-signs")
        axis_signs = np.where(axis_signs >= 0.0, 1.0, -1.0)

    return axis_order, axis_signs, preset


def resolve_force_safety_frame(args: argparse.Namespace) -> str:
    if args.force_safety_frame != "auto":
        return args.force_safety_frame
    if args.force_source == "ur-rtde":
        return "base"
    return "tool"


class CartesianForceSafetyController:
    """
    Task-agnostic force guard for kinesthetic/teleop collection.

    It follows the same idea as the admittance code in ur5moverealrobot3.py:
    excessive Cartesian force/torque is converted into a small compliant retreat
    before the commanded TCP target is sent. Translation and posture are corrected;
    the operator's gripper command is left untouched unless a hard stop is active.
    """

    def __init__(self, args: argparse.Namespace, dt: float) -> None:
        force_threshold = parse_xyz_vector(
            args.force_safety_threshold_n,
            "--force-safety-threshold-n",
            positive=True,
        )
        torque_threshold = parse_xyz_vector(
            args.torque_safety_threshold_nm,
            "--torque-safety-threshold-nm",
            positive=True,
        )
        force_hard_stop = parse_xyz_vector(
            args.force_safety_hard_stop_n,
            "--force-safety-hard-stop-n",
            positive=True,
        )
        torque_hard_stop = parse_xyz_vector(
            args.torque_safety_hard_stop_nm,
            "--torque-safety-hard-stop-nm",
            positive=True,
        )
        self.threshold = np.concatenate([force_threshold, torque_threshold])
        self.hard_stop = np.concatenate([force_hard_stop, torque_hard_stop])
        self.force_frame = resolve_force_safety_frame(args)
        self.axis_order, self.axis_signs, self.axis_preset = resolve_force_safety_axis_mapping(args)
        self.tool_offset_m = float(args.force_torque_tool_offset_m)
        self.inv_mass = np.concatenate(
            [
                np.full(3, float(args.force_safety_inv_mass), dtype=np.float64),
                np.full(3, float(args.torque_safety_inv_inertia), dtype=np.float64),
            ]
        )
        self.damping = np.concatenate(
            [
                np.full(3, float(args.force_safety_damping), dtype=np.float64),
                np.full(3, float(args.torque_safety_damping), dtype=np.float64),
            ]
        )
        self.stiffness = np.concatenate(
            [
                np.full(3, float(args.force_safety_stiffness), dtype=np.float64),
                np.full(3, float(args.torque_safety_stiffness), dtype=np.float64),
            ]
        )
        self.max_vel = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_max_vel)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_max_angular_vel)), dtype=np.float64),
            ]
        )
        self.max_correction = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_max_correction_m)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_max_correction_rad)), dtype=np.float64),
            ]
        )
        self.deadband = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_deadband_n)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_deadband_nm)), dtype=np.float64),
            ]
        )
        self.dt = float(dt)
        self.offset = np.zeros(6, dtype=np.float64)
        self.velocity = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self.offset.fill(0.0)
        self.velocity.fill(0.0)

    def _wrench_in_base(self, force_torque: np.ndarray, current_tcp: np.ndarray) -> np.ndarray:
        raw = np.asarray(force_torque, dtype=np.float64).reshape(-1)
        if raw.size < 6:
            raw = np.pad(raw, (0, 6 - raw.size))
        force_raw = raw[:3]
        torque_raw = raw[3:6]
        force = force_raw[self.axis_order] * self.axis_signs
        if self.axis_preset == "fts300-tool":
            torque = np.asarray(
                [
                    (torque_raw[0] - force_raw[1] * self.tool_offset_m),
                    (torque_raw[1] + self.tool_offset_m * force_raw[0]),
                    -torque_raw[2],
                ],
                dtype=np.float64,
            )
        else:
            torque = torque_raw[self.axis_order] * self.axis_signs

        if self.force_frame == "tool":
            rotation = rotvec_to_matrix(np.asarray(current_tcp, dtype=np.float64)[3:6])
            force = rotation @ force
            torque = rotation @ torque
        return np.concatenate([force, torque])

    def update(
        self,
        force_torque: np.ndarray,
        current_tcp: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        wrench = self._wrench_in_base(force_torque, current_tcp)
        abs_wrench = np.abs(wrench)
        active = abs_wrench > (self.threshold + self.deadband)
        excess = np.zeros(6, dtype=np.float64)
        excess[active] = np.sign(wrench[active]) * (abs_wrench[active] - self.threshold[active])

        # Oppose the measured wrench. The spring term smoothly returns the
        # correction to zero after contact force drops.
        acceleration = (
            -self.inv_mass * excess
            - self.damping * self.velocity
            - self.stiffness * self.offset
        )
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_vel, self.max_vel)
        self.offset += self.velocity * self.dt
        self.offset = np.clip(self.offset, -self.max_correction, self.max_correction)

        hard_stop = bool(np.any(abs_wrench > self.hard_stop))
        return self.offset.copy(), hard_stop, wrench

    def correct_target(
        self,
        current_tcp: np.ndarray,
        target_tcp: np.ndarray,
        force_torque: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        correction, hard_stop, signed_wrench = self.update(force_torque, current_tcp)
        corrected = np.asarray(target_tcp, dtype=np.float64).copy()
        current_tcp = np.asarray(current_tcp, dtype=np.float64)
        if hard_stop:
            corrected[:3] = current_tcp[:3]
            corrected[3:6] = current_tcp[3:6]

        corrected[:3] += correction[:3]
        correction_rotation = rotvec_to_matrix(correction[3:6])
        target_rotation = rotvec_to_matrix(corrected[3:6])
        corrected[3:6] = matrix_to_rotvec(correction_rotation @ target_rotation)
        return corrected, hard_stop, correction, signed_wrench


class ForceReader:
    def __init__(self, args: argparse.Namespace, robot: ZMQClientRobot) -> None:
        self.source = args.force_source
        self._robot = robot
        self._modbus = None
        if self.source == "modbus-serial":
            self._modbus = RobotiqFTS300ModbusSerial(
                port=args.force_serial_port,
                slave_address=args.force_serial_slave_address,
                register_address=args.force_serial_register_address,
                function_code=args.force_serial_function_code,
                baudrate=args.force_serial_baudrate,
                timeout_s=args.force_serial_timeout_s,
                wakeup=args.force_serial_wakeup,
                retries=args.force_serial_retries,
            )
        elif self.source == "modbus-tcp":
            self._modbus = RobotiqFTS300ModbusTCP(
                host=args.force_modbus_host,
                port=args.force_modbus_port,
                unit_id=args.force_modbus_unit_id,
                register_address=args.force_modbus_address,
                function=args.force_modbus_function,
                value_format=args.force_modbus_format,
                word_order=args.force_modbus_word_order,
                scale=args.force_modbus_scale,
                timeout_s=args.force_modbus_timeout_s,
            )

    def read(self) -> np.ndarray:
        if self.source in ("modbus-serial", "modbus-tcp"):
            if self._modbus is None:
                raise RuntimeError("Modbus force reader is not initialized")
            return format_force_torque(self._modbus.read())
        if self.source == "ur-rtde":
            return format_force_torque(self._robot.get_tcp_force())
        raise RuntimeError(f"Unsupported force source: {self.source}")

    def close(self) -> None:
        close = getattr(self._modbus, "close", None)
        if close is not None:
            close()


def calibrate_force_bias(force_reader: ForceReader, fps: int) -> np.ndarray:
    num_samples = max(10, int(FORCE_TARE_SECONDS * fps))
    samples = []
    print(
        "Calibrating FTS-300-S force bias. Keep the end effector still and "
        "unloaded..."
    )
    for _ in range(num_samples):
        samples.append(force_reader.read())
        time.sleep(1.0 / max(1, fps))

    bias = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    values = [float(v) for v in bias]
    print(
        "FTS-300-S bias "
        f"[Fx,Fy,Fz,Tx,Ty,Tz]=[{values[0]: .3f}, {values[1]: .3f}, "
        f"{values[2]: .3f}, {values[3]: .3f}, {values[4]: .3f}, {values[5]: .3f}]"
    )
    return bias


def empty_tactile_frame(max_points: int) -> dict[str, np.ndarray | float | bool]:
    return {
        "displacement": np.zeros((max_points, 3), dtype=np.float32),
        "distributed_force": np.zeros((max_points, 3), dtype=np.float32),
        "wrench": np.zeros(6, dtype=np.float32),
        "timestamp": 0.0,
        "valid": False,
    }


def format_tactile_frame(
    frame: dict[str, np.ndarray | float | bool],
    max_points: int,
) -> dict[str, np.ndarray | float | bool]:
    formatted = empty_tactile_frame(max_points)
    for key in ("displacement", "distributed_force"):
        array = np.asarray(frame[key], dtype=np.float32).reshape(-1, 3)
        count = min(array.shape[0], max_points)
        formatted[key][:count] = array[:count]
    wrench = np.asarray(frame["wrench"], dtype=np.float32).reshape(-1)
    if wrench.size < 6:
        wrench = np.pad(wrench, (0, 6 - wrench.size))
    formatted["wrench"] = wrench[:6].astype(np.float32)
    formatted["timestamp"] = float(frame.get("timestamp", 0.0))
    formatted["valid"] = bool(frame.get("valid", True))
    return formatted


class PaxiniS2716TactileReader:
    """Read Paxini/PX-6AX tactile data over USB serial.

    This mirrors /home/zhou/vla/PX-6AX/USB_UI.py without the Tk UI. The Paxini
    S2716_core returns 3 bytes per taxel for distributed force and 3 bytes for
    resultant force. Values are converted with the same 0.1 N scale.
    """

    def __init__(
        self,
        port: str | None,
        *,
        model: str = PAXINI_DEFAULT_MODEL,
        module_id: str = "02",
        device_addr: str | None = None,
        baudrate: int = 921600,
        timeout_s: float = 0.1,
        read_timeout_s: float = 0.5,
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
        return {
            "displacement": displacement,
            "distributed_force": distributed,
            "wrench": wrench,
            "timestamp": time.time(),
            "valid": distributed.size > 0 or bool(np.any(resultant)),
        }

    def close(self) -> None:
        if getattr(self, "ser", None) is not None and self.ser.is_open:
            self.ser.close()


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + np.sin(theta) * axis_skew
        + (1.0 - np.cos(theta)) * (axis_skew @ axis_skew)
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(theta)
    return axis * theta


def rotation_matrix(angle: float, direction: tuple[float, float, float]) -> np.ndarray:
    return rotvec_to_matrix(np.asarray(direction, dtype=np.float64) * angle)



class KeyboardTeleop:
    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((520, 160))
        pygame.display.set_caption("UR5 RG2 keyboard collection")
        self.gripper_closed = False
        self.screw_direction = 0
        self._paint((80, 80, 80))

    def _paint(self, color: tuple[int, int, int]) -> None:
        self.screen.fill(color)
        self.pygame.display.flip()

    def close(self) -> None:
        self.pygame.quit()

    def poll(self, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, bool, str | None]:
        self.pygame.event.pump()
        event = None
        for pygame_event in self.pygame.event.get():
            if pygame_event.type != self.pygame.KEYDOWN:
                continue
            if pygame_event.key == self.pygame.K_z:
                self.screw_direction = 0
                event = "reset"
            elif pygame_event.key == self.pygame.K_RETURN:
                self.screw_direction = 0
                event = "done"
            elif pygame_event.key == self.pygame.K_ESCAPE:
                self.screw_direction = 0
                event = "quit"
            elif pygame_event.key == self.pygame.K_SPACE:
                self.gripper_closed = not self.gripper_closed
            elif pygame_event.key == self.pygame.K_p:
                event = "start"
            elif pygame_event.key == self.pygame.K_l:
                event = "random_reset_pose"
            elif pygame_event.key == self.pygame.K_o:
                event = "capture_reset_orientation"
            elif pygame_event.key == self.pygame.K_t:
                self.screw_direction = 0 if self.screw_direction == 1 else 1
                print("Screw down enabled" if self.screw_direction else "Screw motion stopped")
            elif pygame_event.key == self.pygame.K_g:
                self.screw_direction = 0 if self.screw_direction == -1 else -1
                print("Screw up enabled" if self.screw_direction else "Screw motion stopped")

        keys = self.pygame.key.get_pressed()
        move_step = args.move_step
        shift_pressed = keys[self.pygame.K_LSHIFT] or keys[self.pygame.K_RSHIFT]
        speed_scale = args.shift_speed_scale if shift_pressed else 1.0
        move_step = args.move_step * speed_scale

        dpos = np.zeros(3, dtype=np.float64)
        drot = np.eye(3)

        if keys[self.pygame.K_s]:
            dpos += np.array([move_step, 0.0, 0.0])
        if keys[self.pygame.K_w]:
            dpos += np.array([-move_step, 0.0, 0.0])
        if keys[self.pygame.K_a]:
            dpos += np.array([0.0, -move_step, 0.0])
        if keys[self.pygame.K_d]:
            dpos += np.array([0.0, move_step, 0.0])
        if keys[self.pygame.K_r]:
            dpos += np.array([0.0, 0.0, move_step])
        if keys[self.pygame.K_f]:
            dpos += np.array([0.0, 0.0, -move_step])

        if keys[self.pygame.K_LEFT]:
            drot = rotation_matrix(args.rot_step, (0.0, 1.0, 0.0))
        if keys[self.pygame.K_RIGHT]:
            drot = rotation_matrix(-args.rot_step, (0.0, 1.0, 0.0))
        if keys[self.pygame.K_DOWN]:
            drot = rotation_matrix(args.rot_step, (1.0, 0.0, 0.0))
        if keys[self.pygame.K_UP]:
            drot = rotation_matrix(-args.rot_step, (1.0, 0.0, 0.0))
        if keys[self.pygame.K_q]:
            drot = rotation_matrix(args.rot_step, (0.0, 0.0, 1.0))
        if keys[self.pygame.K_e]:
            drot = rotation_matrix(-args.rot_step, (0.0, 0.0, 1.0))

        screw_rot_step = args.screw_rot_step
        if args.screw_rpm is not None:
            screw_rot_step = 2.0 * np.pi * float(args.screw_rpm) / (60.0 * args.fps)
        screw_linear_step = args.screw_pitch * screw_rot_step / (2.0 * np.pi)
        if self.screw_direction == 1:
            dpos += np.array([0.0, 0.0, -screw_linear_step])
            drot = rotation_matrix(screw_rot_step, (0.0, 0.0, 1.0))
        elif self.screw_direction == -1:
            dpos += np.array([0.0, 0.0, screw_linear_step])
            drot = rotation_matrix(-screw_rot_step, (0.0, 0.0, 1.0))

        has_motion = bool(np.any(dpos) or not np.allclose(drot, np.eye(3)))
        self._paint((0, 140, 0) if has_motion else (80, 80, 80))
        gripper = 1.0 if self.gripper_closed else 0.0
        return dpos, drot, gripper, has_motion, event



def apply_tcp_delta(tcp_pose: np.ndarray, dpos: np.ndarray, drot: np.ndarray) -> np.ndarray:
    target = np.asarray(tcp_pose, dtype=np.float64).copy()
    target[:3] += dpos
    current_rot = rotvec_to_matrix(target[3:6])
    target[3:6] = matrix_to_rotvec(current_rot @ drot)
    return target


def sample_random_reset_tcp(reset_rotation_vector: np.ndarray) -> np.ndarray:
    target = np.zeros(6, dtype=np.float64)
    target[:3] = RESET_TCP_POSITION + np.random.uniform(
        -RESET_TCP_POSITION_RANDOM_RANGE_M,
        RESET_TCP_POSITION_RANDOM_RANGE_M,
        size=3,
    )
    target[3:6] = np.asarray(reset_rotation_vector, dtype=np.float64)
    return target


def step_toward_joint_state(
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    max_joint_step_rad: float,
) -> tuple[np.ndarray, bool]:
    current = np.asarray(current_joints, dtype=np.float64).copy()
    target = np.asarray(target_joints, dtype=np.float64).reshape(-1)
    next_joints = current.copy()

    arm_delta = target[:6] - current[:6]
    max_delta = float(np.max(np.abs(arm_delta)))
    if max_delta > max_joint_step_rad > 0.0:
        next_joints[:6] = current[:6] + arm_delta / max_delta * max_joint_step_rad
    else:
        next_joints[:6] = target[:6]

    reached = np.max(np.abs(target[:6] - next_joints[:6])) < 1e-3
    return next_joints, reached


class PreviewWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._cv2 = None
        self._warned = False
        if self.enabled:
            import cv2

            self._cv2 = cv2

    def show(self, base_image: np.ndarray, wrist_image: np.ndarray) -> None:
        if not self.enabled or self._cv2 is None:
            return

        preview = np.concatenate([base_image, wrist_image], axis=1)
        try:
            self._cv2.imshow("base | wrist", preview[:, :, ::-1])
            self._cv2.waitKey(1)
        except self._cv2.error as exc:
            self.enabled = False
            if not self._warned:
                self._warned = True
                print(
                    "OpenCV preview is unavailable in this environment; "
                    "continuing collection without preview. "
                    "You can also pass --no-preview."
                )
                print(exc)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    robot = ZMQClientRobot(
        port=args.robot_port,
        host=args.host,
        timeout_ms=args.robot_timeout_ms,
    )
    base_camera = ZMQClientCamera(port=args.base_camera_port, host=args.host)
    wrist_camera = ZMQClientCamera(port=args.wrist_camera_port, host=args.host)
    force_reader = ForceReader(args, robot) if args.collect_force else None
    left_tactile = None
    right_tactile = None
    if args.collect_tactile and args.tactile_source == "tac3d":
        left_tactile = ZMQClientTactile(
            port=args.left_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
        right_tactile = ZMQClientTactile(
            port=args.right_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
    elif args.collect_tactile and args.tactile_source == "paxini":
        left_tactile = PaxiniS2716TactileReader(
            args.paxini_left_port,
            model=args.paxini_model,
            module_id=args.paxini_left_module_id,
            device_addr=args.paxini_left_device_addr,
            baudrate=args.paxini_baudrate,
            timeout_s=args.paxini_timeout_s,
            read_timeout_s=args.paxini_read_timeout_s,
            probe_address=args.paxini_probe_address,
            calibrate_on_start=args.paxini_calibrate_on_start,
        )
        if args.paxini_right_port:
            right_tactile = PaxiniS2716TactileReader(
                args.paxini_right_port,
                model=args.paxini_model,
                module_id=args.paxini_right_module_id,
                device_addr=args.paxini_right_device_addr,
                baudrate=args.paxini_baudrate,
                timeout_s=args.paxini_timeout_s,
                read_timeout_s=args.paxini_read_timeout_s,
                probe_address=args.paxini_probe_address,
                calibrate_on_start=args.paxini_calibrate_on_start,
            )

    try:
        robot_dofs = robot.num_dofs()
        if robot_dofs != 7:
            raise RuntimeError(f"Expected UR5 + RG2 robot with 7 DoF, got {robot_dofs}.")
        base_camera.read((args.image_size, args.image_size))
        wrist_camera.read((args.image_size, args.image_size))
        if args.tactile_source == "tac3d" and left_tactile is not None and right_tactile is not None:
            format_tactile_frame(left_tactile.read(), args.tactile_max_points)
            format_tactile_frame(right_tactile.read(), args.tactile_max_points)
        elif args.tactile_source == "paxini" and left_tactile is not None:
            format_tactile_frame(left_tactile.read(), args.tactile_max_points)
            if right_tactile is not None:
                format_tactile_frame(right_tactile.read(), args.tactile_max_points)
        if force_reader is not None:
            force_reader.read()
    except RuntimeError as exc:
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()
        raise RuntimeError(
            "Failed to connect to or warm up the robot/camera/tactile/force sources. "
            "Start them first with:\n"
            "  python real_ur5rg2/experiments/launch_nodes.py --robot ur "
            "--camera-width 424 --camera-height 240 --camera-fps 15\n"
            "If the servers are already running, check the camera serial numbers "
            "and RealSense USB/frame status. If tactile collection is enabled, "
            "also check --tactile-source, Tac3D SDK/ZMQ ports, or Paxini serial "
            "ports/device addresses. "
            "If force collection is enabled with --force-source modbus-serial, "
            "check the FTS-300-S USB serial port, slave address, register address, "
            "baudrate, and minimalmodbus/pyserial installation. If --force-source "
            "modbus-tcp is used, check IP, port, unit id, register address, "
            "function code, data format, and scale. If --force-source ur-rtde "
            "is used, check that the UR TCP force signal is available through RTDE.\n"
            f"Original error: {exc}"
        ) from exc

    dataset = make_dataset(args)
    store_tactile, store_force = validate_requested_sensor_features(dataset, args)
    period = 1.0 / args.fps
    force_bias = np.zeros(6, dtype=np.float32)
    if store_force and force_reader is not None:
        force_bias = calibrate_force_bias(force_reader, args.fps)
    force_safety = None
    if args.force_safety and store_force and force_reader is not None:
        force_safety = CartesianForceSafetyController(args, dt=period)
        print(
            "Force safety enabled: "
            f"force_threshold={force_safety.threshold[:3].tolist()} N, "
            f"torque_threshold={force_safety.threshold[3:].tolist()} Nm, "
            f"force_hard_stop={force_safety.hard_stop[:3].tolist()} N, "
            f"torque_hard_stop={force_safety.hard_stop[3:].tolist()} Nm, "
            f"frame={force_safety.force_frame}, "
            f"axis_preset={force_safety.axis_preset}, "
            f"axis_order={force_safety.axis_order.tolist()}, "
            f"axis_signs={force_safety.axis_signs.tolist()}, "
            f"max_xyz_correction={force_safety.max_correction[:3].tolist()} m, "
            f"max_rot_correction={force_safety.max_correction[3:].tolist()} rad"
        )
    teleop = KeyboardTeleop()
    preview = PreviewWindow(enabled=not args.no_preview)

    obj_init = np.zeros(9, dtype=np.float32)
    episode_id = 0
    recording = False
    frames_in_episode = 0
    next_tick = time.time()
    reset_target_joints = None
    reset_rotation_vector = RESET_TCP_ROTATION_VECTOR.copy()

    print("Keyboard teleop controls:")
    print("  W/S/A/D/R/F: translate TCP")
    print("  Arrow keys + Q/E: rotate TCP")
    print("  T/G: toggle screw down/up; press the active key again to stop")
    print("  P: start recording")
    print("  L: move to randomized reset pose through IK")
    print("  O: capture current TCP orientation for L reset")
    print("  SPACE: toggle RG2, Z: clear current episode, ENTER: save episode, ESC: exit")

    try:
        while episode_id < args.num_episodes:
            now = time.time()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue
            next_tick = now + period

            tcp_pose = robot.get_tcp_pose()
            obs_before = robot.get_observations()
            current_joints = np.asarray(obs_before["joint_positions"], dtype=np.float32)

            dpos, drot, gripper, has_motion, event = teleop.poll(args)

            # if recording:
                # print(
                #     "TCP pose "
                #     f"[x,y,z,rx,ry,rz]=[{tcp_pose[0]: .4f}, {tcp_pose[1]: .4f}, "
                #     f"{tcp_pose[2]: .4f}, {tcp_pose[3]: .3f}, {tcp_pose[4]: .3f}, {tcp_pose[5]: .3f}]"
                # )

            if event == "reset":
                reset_target_joints = None
                reset_episode_buffer(dataset)
                if force_safety is not None:
                    force_safety.reset()
                recording = False
                frames_in_episode = 0
                print("Reset current episode buffer")
                continue
            if event == "done":
                reset_target_joints = None
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} ({frames_in_episode} frames)")
                    episode_id += 1
                if force_safety is not None:
                    force_safety.reset()
                recording = False
                frames_in_episode = 0
                continue
            # Start recording immediately when P is pressed, instead of waiting for motion. This allows collecting "zero" episodes with no motion, which can be useful for certain applications.
            if event == "start":
                if not recording:
                    recording = True
                    frames_in_episode = 0
                    reset_episode_buffer(dataset)
                    if force_safety is not None:
                        force_safety.reset()
                    print(f"Start recording episode {episode_id}")
            if event == "quit":
                reset_target_joints = None
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} before exit ({frames_in_episode} frames)")
                break

            if event == "capture_reset_orientation":
                reset_rotation_vector = np.asarray(tcp_pose[3:6], dtype=np.float64).copy()
                print(
                    "Captured reset TCP orientation "
                    f"[rx,ry,rz]=[{reset_rotation_vector[0]: .6f}, "
                    f"{reset_rotation_vector[1]: .6f}, {reset_rotation_vector[2]: .6f}]"
                )
                continue

            if event == "random_reset_pose":
                reset_tcp = sample_random_reset_tcp(reset_rotation_vector)
                try:
                    reset_target_joints = np.asarray(
                        robot.inverse_kinematics(reset_tcp, qnear=current_joints[:6]),
                        dtype=np.float64,
                    )
                    reset_target_joints[5] -= 0.5 * np.pi
                except RuntimeError as exc:
                    print(
                        "Reset IK failed. Restart launch_nodes.py so the robot "
                        f"server exposes inverse_kinematics. Error: {exc}"
                    )
                    continue
                if force_safety is not None:
                    force_safety.reset()
                print(
                    "Move to randomized reset pose through IK "
                    f"tcp=[{reset_tcp[0]: .4f}, {reset_tcp[1]: .4f}, "
                    f"{reset_tcp[2]: .4f}, {reset_tcp[3]: .3f}, "
                    f"{reset_tcp[4]: .3f}, {reset_tcp[5]: .3f}] "
                    f"q={np.array2string(reset_target_joints, precision=4)}"
                )

            reset_pose_requested = reset_target_joints is not None
            if reset_pose_requested:
                target_joints, reset_reached = step_toward_joint_state(
                    current_joints,
                    reset_target_joints,
                    abs(float(args.reset_max_joint_step_rad)),
                )
                target_joints[-1] = gripper
                if reset_reached:
                    reset_target_joints = None
                    actual_tcp = robot.get_tcp_pose()
                    print(
                        "Reached randomized reset IK joint target; actual TCP "
                        f"[x,y,z,rx,ry,rz]=[{actual_tcp[0]: .4f}, {actual_tcp[1]: .4f}, "
                        f"{actual_tcp[2]: .4f}, {actual_tcp[3]: .3f}, "
                        f"{actual_tcp[4]: .3f}, {actual_tcp[5]: .3f}]"
                    )
            else:
                target_tcp = apply_tcp_delta(tcp_pose, dpos, drot)

            force_torque = np.zeros(6, dtype=np.float32)
            if store_force and force_reader is not None:
                try:
                    raw_force_torque = force_reader.read()
                    force_torque = (raw_force_torque - force_bias).astype(np.float32)
                except RuntimeError as exc:
                    print(f"Force/torque frame skipped: {exc}")
                    continue

            safety_active = False
            safety_hard_stop = False
            safety_correction = np.zeros(6, dtype=np.float64)
            if force_safety is not None and not reset_pose_requested:
                target_tcp, safety_hard_stop, safety_correction, signed_wrench = (
                    force_safety.correct_target(tcp_pose, target_tcp, force_torque)
                )
                safety_active = bool(np.any(np.abs(safety_correction) > 1e-6))
                if safety_active or safety_hard_stop:
                    print(
                        "Force safety "
                        f"{'HARD ' if safety_hard_stop else ''}"
                        f"force_xyz={np.array2string(signed_wrench[:3], precision=3)} "
                        f"torque_xyz={np.array2string(signed_wrench[3:], precision=3)} "
                        f"pos_corr_m={np.array2string(safety_correction[:3], precision=5)} "
                        f"rot_corr_rad={np.array2string(safety_correction[3:], precision=5)}"
                    )

            if (
                reset_pose_requested
                or has_motion
                or safety_active
                or abs(float(current_joints[-1]) - gripper) > 0.03
            ):
                if not args.no_command:
                    if reset_pose_requested:
                        robot.command_joint_state(target_joints)
                    elif has_motion or safety_active:
                        robot.command_tcp_pose(target_tcp)
                    if not safety_hard_stop:
                        robot.command_gripper(gripper, force=args.gripper_force)

            try:
                base_image, _ = base_camera.read((args.image_size, args.image_size))
                wrist_image, _ = wrist_camera.read((args.image_size, args.image_size))
            except RuntimeError as exc:
                print(f"Camera frame skipped: {exc}")
                continue
            base_image = resize_rgb(base_image, args.image_size)
            wrist_image = resize_rgb(wrist_image, args.image_size)

            if not store_tactile or left_tactile is None:
                left_tactile_frame = empty_tactile_frame(args.tactile_max_points)
                right_tactile_frame = empty_tactile_frame(args.tactile_max_points)
            else:
                try:
                    left_tactile_frame = format_tactile_frame(
                        left_tactile.read(),
                        args.tactile_max_points,
                    )
                    if right_tactile is None:
                        right_tactile_frame = empty_tactile_frame(args.tactile_max_points)
                    else:
                        right_tactile_frame = format_tactile_frame(
                            right_tactile.read(),
                            args.tactile_max_points,
                        )
                except RuntimeError as exc:
                    print(f"Tactile frame skipped: {exc}")
                    continue

            obs_after = robot.get_observations()
            state = np.asarray(obs_after["joint_positions"], dtype=np.float32)
            action = state.copy()
            action[-1] = float(state[-1]) if safety_hard_stop else gripper

            
            # Recording now starts only when P is pressed.

            # if not recording and (
            #     has_motion
            #     or safety_active
            #     or abs(float(current_joints[-1]) - gripper) > 0.03
            # ):
            #     recording = True
            #     frames_in_episode = 0
            #     reset_episode_buffer(dataset)
            #     print(f"Start recording episode {episode_id}")

            if recording:
                frame = {
                    "observation.image": base_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": state[:6].astype(np.float32),
                    "action": action.astype(np.float32),
                    "obj_init": obj_init,
                }
                if store_tactile:
                    frame.update(
                        {
                            "observation.tactile_left.displacement": left_tactile_frame[
                                "displacement"
                            ],
                            "observation.tactile_left.distributed_force": left_tactile_frame[
                                "distributed_force"
                            ],
                            "observation.tactile_left.wrench": left_tactile_frame[
                                "wrench"
                            ],
                            "observation.tactile_right.displacement": right_tactile_frame[
                                "displacement"
                            ],
                            "observation.tactile_right.distributed_force": right_tactile_frame[
                                "distributed_force"
                            ],
                            "observation.tactile_right.wrench": right_tactile_frame[
                                "wrench"
                            ],
                            "observation.tactile.timestamp": np.array(
                                [
                                    left_tactile_frame["timestamp"],
                                    right_tactile_frame["timestamp"],
                                ],
                                dtype=np.float64,
                            ),
                        }
                    )
                if store_force:
                    frame[FORCE_TORQUE_KEY] = force_torque
                    print_every = max(1, int(args.force_print_every))
                    # 不打印力
                    # if frames_in_episode % print_every == 0:
                    #     print_force_torque(force_torque)

                dataset.add_frame(frame, task=args.task)
                frames_in_episode += 1

            preview.show(base_image, wrist_image)
    finally:
        preview.close()
        teleop.close()
        if force_reader is not None:
            force_reader.close()
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()


if __name__ == "__main__":
    main()
