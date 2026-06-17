from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("/home/zhou/vla/force-vla/lerobot-mujoco-vla")
for path in (PROJECT_ROOT, REFERENCE_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_POLICY_PATHS = (
    PROJECT_ROOT / "gufic_env/flow_matching/checkpoints_smolvla_real_ur5/checkpoints/last/pretrained_model",
    PROJECT_ROOT / "gufic_env/flow_matching/checkpoints_smolvla_v_position/020000/pretrained_model",
    PROJECT_ROOT / "gufic_env/flow_matching/checkpoints_smolvla_v2/020000/pretrained_model",
)
DEFAULT_VLM_MODEL_PATHS = (
    Path(
        "/home/zhou/.cache/huggingface/hub/"
        "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    ),
    Path(
        "/root/autodl-tmp/hub/"
        "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    ),
)

FORCE_TARE_SECONDS = 2.0
FORCE_KEYS = (
    "observation.force_torque",
    "observation.force",
)
TACTILE_KEYS = (
    "observation.tactile_left.displacement",
    "observation.tactile_left.distributed_force",
    "observation.tactile_left.wrench",
    "observation.tactile_right.displacement",
    "observation.tactile_right.distributed_force",
    "observation.tactile_right.wrench",
    "observation.tactile.timestamp",
)
torch = None


def ensure_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a SmolVLA policy on the real UR5 + RG2 ZMQ nodes."
    )
    parser.add_argument("--repo-id", default="ur5_rg2_real_smolvla")
    parser.add_argument(
        "--root",
        default="./real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force",
        help="LeRobot dataset root used for metadata/stats.",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--vlm-model-name", default=None)
    parser.add_argument("--task", default="Insert the bolt into the nut.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--robot-timeout-ms", type=int, default=3000)
    parser.add_argument("--wrist-camera-port", type=int, default=5000)
    parser.add_argument("--base-camera-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-ms", type=int, default=20000)
    parser.add_argument(
        "--collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read Tac3D tactile fields if the loaded policy has tactile inputs.",
    )
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-max-points", type=int, default=400)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument(
        "--no-tactile",
        action="store_false",
        dest="collect_tactile",
        help="Alias for --no-collect-tactile.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without commanding the robot.")
    parser.add_argument("--no-command", action="store_true", help="Alias for --dry-run.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--action-lowpass-alpha", type=float, default=1.0)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.02,
        help="Maximum absolute joint change per inference step in radians.",
    )
    parser.add_argument(
        "--max-startup-joint-delta",
        type=float,
        default=0.005,
        help="Conservative joint delta used for the first startup steps.",
    )
    parser.add_argument("--startup-steps", type=int, default=50)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument(
        "--command-gripper-separately",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also send the 7th action through command_gripper after command_joint_state.",
    )
    parser.add_argument(
        "--collect-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read force/torque and feed it if the policy has a force input.",
    )
    parser.add_argument(
        "--force-source",
        choices=("modbus-serial", "modbus-tcp", "ur-rtde"),
        default="modbus-serial",
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-slave-address", type=int, default=9)
    parser.add_argument("--force-serial-register-address", type=int, default=180)
    parser.add_argument("--force-serial-function-code", type=int, default=3)
    parser.add_argument("--force-serial-baudrate", type=int, default=19200)
    parser.add_argument("--force-serial-timeout-s", type=float, default=0.1)
    parser.add_argument("--force-serial-retries", type=int, default=5)
    parser.add_argument("--force-serial-wakeup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-modbus-host", default="192.168.1.1")
    parser.add_argument("--force-modbus-port", type=int, default=502)
    parser.add_argument("--force-modbus-unit-id", type=int, default=9)
    parser.add_argument("--force-modbus-address", type=int, default=0)
    parser.add_argument("--force-modbus-function", choices=("holding", "input"), default="input")
    parser.add_argument("--force-modbus-format", choices=("int16", "int32", "float32"), default="int32")
    parser.add_argument("--force-modbus-word-order", choices=("big", "little"), default="big")
    parser.add_argument("--force-modbus-scale", type=float, default=1.0)
    parser.add_argument("--force-modbus-timeout-s", type=float, default=1.0)
    return parser.parse_args()


def first_existing_path(paths) -> str | None:
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def resolve_policy_path(policy_path: str | None) -> str:
    if policy_path:
        return str(policy_path)
    env_path = os.environ.get("SMOLVLA_POLICY_PATH")
    if env_path:
        return env_path
    path = first_existing_path(DEFAULT_POLICY_PATHS)
    if path:
        return path
    raise FileNotFoundError(
        "No SmolVLA policy checkpoint found. Pass --policy-path or set SMOLVLA_POLICY_PATH."
    )


def resolve_vlm_model_name(vlm_model_name: str | None) -> str:
    if vlm_model_name:
        if not Path(vlm_model_name).exists():
            raise FileNotFoundError(f"--vlm-model-name path does not exist: {vlm_model_name}")
        return str(vlm_model_name)
    env_path = os.environ.get("SMOLVLA_VLM_MODEL_NAME")
    if env_path:
        if not Path(env_path).exists():
            raise FileNotFoundError(f"SMOLVLA_VLM_MODEL_NAME path does not exist: {env_path}")
        return env_path
    path = first_existing_path(DEFAULT_VLM_MODEL_PATHS)
    if path:
        return path
    raise FileNotFoundError(
        "SmolVLA needs a local SmolVLM snapshot. Pass --vlm-model-name or set SMOLVLA_VLM_MODEL_NAME."
    )


def load_smolvla_config(policy_path: str, device: str, vlm_model_name: str):
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig

    try:
        config = PreTrainedConfig.from_pretrained(str(policy_path))
    except Exception as exc:
        train_config_path = Path(policy_path) / "train_config.json"
        if not train_config_path.exists():
            raise RuntimeError(
                f"Failed to parse SmolVLA config from {policy_path}, and train_config.json was not found."
            ) from exc

        with open(train_config_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        if policy_config.get("type") != "smolvla":
            raise RuntimeError(
                f"train_config.json does not contain a SmolVLA policy config: {train_config_path}"
            ) from exc

        valid_fields = {field.name for field in fields(SmolVLAConfig)}
        skip_fields = {"type", "input_features", "output_features", "normalization_mapping"}
        kwargs = {
            key: value
            for key, value in policy_config.items()
            if key in valid_fields and key not in skip_fields
        }
        config = SmolVLAConfig(**kwargs)

    config.vlm_model_name = vlm_model_name
    config.device = device
    return config


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == (image_size, image_size):
        return image.copy()
    return np.asarray(Image.fromarray(image).resize((image_size, image_size)), dtype=np.uint8)


def rgb_to_tensor(image: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    torch_module = ensure_torch()
    image = resize_rgb(image, image_size)
    tensor = torch_module.from_numpy(image).to(device=device, dtype=torch_module.float32) / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


def format_force_torque(force_torque: np.ndarray) -> np.ndarray:
    array = np.asarray(force_torque, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


def empty_tactile_frame(max_points: int) -> dict[str, np.ndarray | float | bool]:
    return {
        "displacement": np.zeros((max_points, 3), dtype=np.float32),
        "distributed_force": np.zeros((max_points, 3), dtype=np.float32),
        "wrench": np.zeros(6, dtype=np.float32),
        "timestamp": 0.0,
        "valid": False,
    }


def format_tactile_frame(
    frame: dict,
    max_points: int,
) -> dict[str, np.ndarray | float | bool]:
    formatted = empty_tactile_frame(max_points)
    for key in ("displacement", "distributed_force"):
        if key not in frame:
            continue
        array = np.asarray(frame[key], dtype=np.float32).reshape(-1, 3)
        count = min(array.shape[0], max_points)
        formatted[key][:count] = array[:count]

    if "wrench" in frame:
        wrench = np.asarray(frame["wrench"], dtype=np.float32).reshape(-1)
        if wrench.size < 6:
            wrench = np.pad(wrench, (0, 6 - wrench.size))
        formatted["wrench"] = wrench[:6].astype(np.float32)

    formatted["timestamp"] = float(frame.get("timestamp", 0.0))
    formatted["valid"] = bool(frame.get("valid", True))
    return formatted


def empty_tactile_inputs(max_points: int) -> dict[str, np.ndarray]:
    left = empty_tactile_frame(max_points)
    right = empty_tactile_frame(max_points)
    return tactile_inputs_from_frames(left, right)


def tactile_inputs_from_frames(left_frame: dict, right_frame: dict) -> dict[str, np.ndarray]:
    return {
        "observation.tactile_left.displacement": np.asarray(
            left_frame["displacement"],
            dtype=np.float32,
        ),
        "observation.tactile_left.distributed_force": np.asarray(
            left_frame["distributed_force"],
            dtype=np.float32,
        ),
        "observation.tactile_left.wrench": format_force_torque(left_frame["wrench"]),
        "observation.tactile_right.displacement": np.asarray(
            right_frame["displacement"],
            dtype=np.float32,
        ),
        "observation.tactile_right.distributed_force": np.asarray(
            right_frame["distributed_force"],
            dtype=np.float32,
        ),
        "observation.tactile_right.wrench": format_force_torque(right_frame["wrench"]),
        "observation.tactile.timestamp": np.asarray(
            [left_frame.get("timestamp", 0.0), right_frame.get("timestamp", 0.0)],
            dtype=np.float64,
        ),
    }


class ForceReader:
    def __init__(self, args: argparse.Namespace, robot) -> None:
        from gello.force.robotiq_fts300 import (
            RobotiqFTS300ModbusSerial,
            RobotiqFTS300ModbusTCP,
        )

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


def calibrate_force_bias(force_reader: ForceReader, fps: float) -> np.ndarray:
    num_samples = max(10, int(FORCE_TARE_SECONDS * fps))
    samples = []
    print("Calibrating force bias. Keep the end effector still and unloaded...")
    for _ in range(num_samples):
        samples.append(force_reader.read())
        time.sleep(1.0 / max(1.0, fps))
    bias = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    print("Force bias [Fx,Fy,Fz,Tx,Ty,Tz] =", np.array2string(bias, precision=4))
    return bias


class RealUR5SmolVLAInfer:
    def __init__(self, args: argparse.Namespace) -> None:
        from lerobot.common.constants import OBS_STATE
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.datasets.utils import dataset_to_policy_features
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
        from lerobot.configs.types import FeatureType

        torch_module = ensure_torch()
        self.args = args
        self.device = args.device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        self.policy_path = resolve_policy_path(args.policy_path)
        self.vlm_model_name = resolve_vlm_model_name(args.vlm_model_name)

        metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
        policy_features = dataset_to_policy_features(metadata.features)
        config = load_smolvla_config(self.policy_path, self.device, self.vlm_model_name)
        config.input_features = {
            key: ft for key, ft in policy_features.items() if ft.type is not FeatureType.ACTION
        }
        config.output_features = {
            key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION
        }

        self.policy = SmolVLAPolicy.from_pretrained(
            self.policy_path,
            config=config,
            dataset_stats=metadata.stats,
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

        state_keys = [
            key for key, ft in self.policy.config.input_features.items() if ft.type is FeatureType.STATE
        ]
        self.force_keys = [key for key in FORCE_KEYS if key in state_keys]
        self.tactile_keys = [key for key in TACTILE_KEYS if key in state_keys]
        non_robot_state_keys = set(self.force_keys) | set(self.tactile_keys)
        non_force_state_keys = [key for key in state_keys if key not in non_robot_state_keys]
        if OBS_STATE in non_force_state_keys:
            self.state_key = OBS_STATE
        elif "observation.state" in non_force_state_keys:
            self.state_key = "observation.state"
        elif non_force_state_keys:
            self.state_key = non_force_state_keys[0]
        else:
            raise ValueError("SmolVLA policy has no robot state input feature.")

        if self.state_key != OBS_STATE:

            def prepare_state(policy_self, batch):
                state = batch[self.state_key]
                state = state[:, -1, :] if state.ndim > 2 else state
                return pad_vector(state, policy_self.config.max_state_dim)

            def prepare_language(policy_self, batch):
                device = batch[self.state_key].device
                tasks = batch["task"]
                if len(tasks) == 1:
                    tasks = [tasks[0] for _ in range(batch[self.state_key].shape[0])]
                tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
                tokenized_prompt = policy_self.language_tokenizer.__call__(
                    tasks,
                    padding=policy_self.config.pad_language_to,
                    padding_side="right",
                    max_length=policy_self.config.tokenizer_max_length,
                    return_tensors="pt",
                )
                lang_tokens = tokenized_prompt["input_ids"].to(device=device)
                lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch_module.bool)
                return lang_tokens, lang_masks

            self.policy.prepare_state = types.MethodType(prepare_state, self.policy)
            self.policy.prepare_language = types.MethodType(prepare_language, self.policy)

        action_features = list(self.policy.config.output_features.values())
        if not action_features or int(action_features[0].shape[0]) < 7:
            raise ValueError("Expected a SmolVLA action feature with at least 7 dimensions.")

        print(f"Loaded policy: {self.policy_path}")
        print(f"Dataset metadata: repo_id={args.repo_id}, root={args.root}")
        print(f"Robot state key: {self.state_key}; force keys: {self.force_keys or 'none'}")
        print(f"Tactile keys: {self.tactile_keys or 'none'}")

    def _select_image_for_key(self, key: str, wrist_image: np.ndarray, base_image: np.ndarray) -> np.ndarray:
        return wrist_image if "wrist" in key else base_image

    def build_batch(
        self,
        base_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_positions: np.ndarray,
        force_torque: np.ndarray,
        tactile_inputs: dict[str, np.ndarray] | None,
    ) -> dict:
        from lerobot.common.constants import OBS_STATE

        batch = {"task": [self.args.task]}
        for key in self.policy.config.image_features:
            image = self._select_image_for_key(key, wrist_image, base_image)
            batch[key] = rgb_to_tensor(image, self.args.image_size, self.device)

        state_dim = int(self.policy.config.input_features[self.state_key].shape[0])
        state = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if state.size < state_dim:
            state = np.pad(state, (0, state_dim - state.size))
        state = state[:state_dim].astype(np.float32)
        torch_module = ensure_torch()
        state_t = torch_module.from_numpy(state[None, :]).to(self.device, dtype=torch_module.float32)
        batch[self.state_key] = state_t
        if OBS_STATE not in batch:
            batch[OBS_STATE] = state_t

        force = format_force_torque(force_torque)
        for key in self.force_keys:
            batch[key] = torch_module.from_numpy(force[None, :]).to(
                self.device,
                dtype=torch_module.float32,
            )

        tactile_inputs = tactile_inputs or {}
        for key in self.tactile_keys:
            expected_shape = tuple(int(dim) for dim in self.policy.config.input_features[key].shape)
            raw_value = tactile_inputs.get(key)
            if raw_value is None:
                value = np.zeros(expected_shape, dtype=np.float32)
            else:
                value = np.asarray(raw_value, dtype=np.float32)
                target = np.zeros(expected_shape, dtype=np.float32)
                source = value.reshape((-1,) + expected_shape[-1:]) if len(expected_shape) == 2 else value.reshape(-1)
                if len(expected_shape) == 2:
                    count = min(source.shape[0], expected_shape[0])
                    cols = min(source.shape[1], expected_shape[1])
                    target[:count, :cols] = source[:count, :cols]
                else:
                    flat = source.reshape(-1)
                    count = min(flat.size, target.size)
                    target.reshape(-1)[:count] = flat[:count]
                value = target
            batch[key] = torch_module.from_numpy(value[None, ...]).to(
                self.device,
                dtype=torch_module.float32,
            )
        return batch

    def predict_action(
        self,
        base_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_positions: np.ndarray,
        force_torque: np.ndarray,
        tactile_inputs: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        batch = self.build_batch(
            base_image,
            wrist_image,
            joint_positions,
            force_torque,
            tactile_inputs,
        )
        torch_module = ensure_torch()
        with torch_module.no_grad():
            action = self.policy.select_action(batch)
        action = action[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        return action[:7]


class PreviewWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._cv2 = None
        if enabled:
            import cv2

            self._cv2 = cv2

    def show(self, base_image: np.ndarray, wrist_image: np.ndarray) -> None:
        if not self.enabled or self._cv2 is None:
            return
        preview = np.concatenate([base_image, wrist_image], axis=1)
        self._cv2.imshow("base | wrist", preview[:, :, ::-1])
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()


def clip_and_filter_action(
    raw_action: np.ndarray,
    current_joints: np.ndarray,
    previous_action: np.ndarray | None,
    args: argparse.Namespace,
    step: int,
) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(7).copy()
    current = np.asarray(current_joints, dtype=np.float32).reshape(-1)
    if current.size >= 6:
        max_delta = args.max_startup_joint_delta if step < args.startup_steps else args.max_joint_delta
        delta = np.clip(action[:6] - current[:6], -max_delta, max_delta)
        action[:6] = current[:6] + delta
    action[6] = float(np.clip(action[6], 0.0, 1.0))

    alpha = float(np.clip(args.action_lowpass_alpha, 0.0, 1.0))
    if previous_action is not None and alpha < 1.0:
        action = alpha * action + (1.0 - alpha) * previous_action
    return action.astype(np.float32)


def main() -> None:
    from gello.zmq_core.camera_node import ZMQClientCamera
    from gello.zmq_core.robot_node import ZMQClientRobot
    from gello.zmq_core.tactile_node import ZMQClientTactile

    args = parse_args()
    args.dry_run = args.dry_run or args.no_command
    infer = RealUR5SmolVLAInfer(args)

    robot = ZMQClientRobot(port=args.robot_port, host=args.host, timeout_ms=args.robot_timeout_ms)
    base_camera = ZMQClientCamera(
        port=args.base_camera_port,
        host=args.host,
        timeout_ms=args.camera_timeout_ms,
    )
    wrist_camera = ZMQClientCamera(
        port=args.wrist_camera_port,
        host=args.host,
        timeout_ms=args.camera_timeout_ms,
    )
    force_reader = ForceReader(args, robot) if args.collect_force else None
    left_tactile = None
    right_tactile = None
    use_tactile = bool(args.collect_tactile and infer.tactile_keys)
    if use_tactile:
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
    preview = PreviewWindow(args.preview)

    force_bias = np.zeros(6, dtype=np.float32)
    previous_action = None
    period = 1.0 / max(1.0, args.fps)
    next_tick = time.time()

    try:
        robot_dofs = robot.num_dofs()
        if robot_dofs != 7:
            raise RuntimeError(f"Expected UR5 + RG2 robot with 7 DoF, got {robot_dofs}.")
        base_camera.read((args.image_size, args.image_size))
        wrist_camera.read((args.image_size, args.image_size))
        if force_reader is not None:
            force_reader.read()
            force_bias = calibrate_force_bias(force_reader, args.fps)
        if left_tactile is not None and right_tactile is not None:
            format_tactile_frame(left_tactile.read(), args.tactile_max_points)
            format_tactile_frame(right_tactile.read(), args.tactile_max_points)

        print("Starting SmolVLA real UR5 inference. Press Ctrl-C to stop.")
        step = 0
        while args.max_steps <= 0 or step < args.max_steps:
            now = time.time()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue
            loop_start = time.time()
            next_tick = loop_start + period

            observations = robot.get_observations()
            joints = np.asarray(observations["joint_positions"], dtype=np.float32)
            base_image, _ = base_camera.read((args.image_size, args.image_size))
            wrist_image, _ = wrist_camera.read((args.image_size, args.image_size))
            base_image = resize_rgb(base_image, args.image_size)
            wrist_image = resize_rgb(wrist_image, args.image_size)

            force_torque = np.zeros(6, dtype=np.float32)
            if force_reader is not None:
                force_torque = (force_reader.read() - force_bias).astype(np.float32)

            tactile_inputs = empty_tactile_inputs(args.tactile_max_points)
            if left_tactile is not None and right_tactile is not None:
                try:
                    left_frame = format_tactile_frame(left_tactile.read(), args.tactile_max_points)
                    right_frame = format_tactile_frame(right_tactile.read(), args.tactile_max_points)
                    tactile_inputs = tactile_inputs_from_frames(left_frame, right_frame)
                except RuntimeError as exc:
                    print(f"Tactile frame skipped: {exc}")
                    continue

            raw_action = infer.predict_action(
                base_image,
                wrist_image,
                joints[:6],
                force_torque,
                tactile_inputs,
            )
            action = clip_and_filter_action(raw_action, joints, previous_action, args, step)
            previous_action = action.copy()

            if not args.dry_run:
                robot.command_joint_state(action)
                if args.command_gripper_separately:
                    robot.command_gripper(float(action[6]), force=args.gripper_force)

            if step % max(1, args.print_every) == 0:
                elapsed = time.time() - loop_start
                print(
                    f"step={step:05d} elapsed={elapsed:.3f}s "
                    f"q_cmd={np.array2string(action[:6], precision=4)} "
                    f"gripper={action[6]:.3f}"
                )
            preview.show(base_image, wrist_image)
            step += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        preview.close()
        if force_reader is not None:
            force_reader.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()
        robot.close()


if __name__ == "__main__":
    main()
