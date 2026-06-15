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
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.tactile_node import ZMQClientTactile


TACTILE_FIELD_KEYS = (
    "observation.tactile_left.displacement",
    "observation.tactile_left.distributed_force",
    "observation.tactile_right.displacement",
    "observation.tactile_right.distributed_force",
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
    parser.add_argument("--root", default="./real_ur5rg2/data/ur5_rg2_real_smolvla_dataset")
    parser.add_argument("--task", default="Insert the bolt into the nut.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--wrist-camera-port", type=int, default=5000)
    parser.add_argument("--base-camera-port", type=int, default=5001)
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-max-points", type=int, default=400)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gripper-force", type=float, default=10.0)
    parser.add_argument("--move-step", type=float, default=0.0005)
    parser.add_argument("--rot-step", type=float, default=0.03)
    parser.add_argument("--screw-pitch", type=float, default=0.0025)
    parser.add_argument("--screw-rot-step", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-command", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-tactile", action="store_true")
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
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        robot_type="ur5_rg2",
        fps=args.fps,
        features={
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
        },
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
                event = "reset"
            elif pygame_event.key == self.pygame.K_RETURN:
                event = "done"
            elif pygame_event.key == self.pygame.K_ESCAPE:
                event = "quit"
            elif pygame_event.key == self.pygame.K_SPACE:
                self.gripper_closed = not self.gripper_closed

        keys = self.pygame.key.get_pressed()
        move_step = args.move_step
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

        screw_linear_step = args.screw_pitch * args.screw_rot_step / (2 * np.pi)
        if keys[self.pygame.K_t]:
            dpos += np.array([0.0, 0.0, -screw_linear_step])
            drot = rotation_matrix(-args.screw_rot_step, (0.0, 1.0, 0.0))
        if keys[self.pygame.K_g]:
            dpos += np.array([0.0, 0.0, screw_linear_step])
            drot = rotation_matrix(args.screw_rot_step, (0.0, 1.0, 0.0))

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

    robot = ZMQClientRobot(port=args.robot_port, host=args.host)
    base_camera = ZMQClientCamera(port=args.base_camera_port, host=args.host)
    wrist_camera = ZMQClientCamera(port=args.wrist_camera_port, host=args.host)
    left_tactile = None
    right_tactile = None
    if not args.no_tactile:
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

    try:
        robot_dofs = robot.num_dofs()
        if robot_dofs != 7:
            raise RuntimeError(f"Expected UR5 + RG2 robot with 7 DoF, got {robot_dofs}.")
        base_camera.read((args.image_size, args.image_size))
        wrist_camera.read((args.image_size, args.image_size))
        if left_tactile is not None and right_tactile is not None:
            format_tactile_frame(left_tactile.read(), args.tactile_max_points)
            format_tactile_frame(right_tactile.read(), args.tactile_max_points)
    except RuntimeError as exc:
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()
        raise RuntimeError(
            "Failed to connect to or warm up the robot/camera/tactile ZMQ servers. "
            "Start them first with:\n"
            "  python real_ur5rg2/experiments/launch_nodes.py --robot ur "
            "--camera-width 424 --camera-height 240 --camera-fps 15\n"
            "If the servers are already running, check the camera serial numbers "
            "and RealSense USB/frame status. If tactile collection is enabled, "
            "also check the Tac3D SDK path, sensor IDs, and tactile ZMQ ports.\n"
            f"Original error: {exc}"
        ) from exc

    dataset = make_dataset(args)
    teleop = KeyboardTeleop()
    preview = PreviewWindow(enabled=not args.no_preview)

    obj_init = np.zeros(9, dtype=np.float32)
    episode_id = 0
    recording = False
    frames_in_episode = 0
    period = 1.0 / args.fps
    next_tick = time.time()

    print("Keyboard teleop controls:")
    print("  W/S/A/D/R/F: translate TCP")
    print("  Arrow keys + Q/E: rotate TCP")
    print("  T/G: screw motion")
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

            if event == "reset":
                reset_episode_buffer(dataset)
                recording = False
                frames_in_episode = 0
                print("Reset current episode buffer")
                continue
            if event == "done":
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} ({frames_in_episode} frames)")
                    episode_id += 1
                recording = False
                frames_in_episode = 0
                continue
            if event == "quit":
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} before exit ({frames_in_episode} frames)")
                break

            target_tcp = apply_tcp_delta(tcp_pose, dpos, drot)

            if has_motion or abs(float(current_joints[-1]) - gripper) > 0.03:
                if not args.no_command:
                    if has_motion:
                        robot.command_tcp_pose(target_tcp)
                    robot.command_gripper(gripper, force=args.gripper_force)

            try:
                base_image, _ = base_camera.read((args.image_size, args.image_size))
                wrist_image, _ = wrist_camera.read((args.image_size, args.image_size))
            except RuntimeError as exc:
                print(f"Camera frame skipped: {exc}")
                continue
            base_image = resize_rgb(base_image, args.image_size)
            wrist_image = resize_rgb(wrist_image, args.image_size)

            if left_tactile is None or right_tactile is None:
                left_tactile_frame = empty_tactile_frame(args.tactile_max_points)
                right_tactile_frame = empty_tactile_frame(args.tactile_max_points)
            else:
                try:
                    left_tactile_frame = format_tactile_frame(
                        left_tactile.read(),
                        args.tactile_max_points,
                    )
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
            action[-1] = gripper

            if not recording and (has_motion or abs(float(current_joints[-1]) - gripper) > 0.03):
                recording = True
                frames_in_episode = 0
                reset_episode_buffer(dataset)
                print(f"Start recording episode {episode_id}")

            if recording:
                dataset.add_frame(
                    {
                        "observation.image": base_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": state[:6].astype(np.float32),
                        "observation.tactile_left.displacement": left_tactile_frame[
                            "displacement"
                        ],
                        "observation.tactile_left.distributed_force": left_tactile_frame[
                            "distributed_force"
                        ],
                        "observation.tactile_left.wrench": left_tactile_frame["wrench"],
                        "observation.tactile_right.displacement": right_tactile_frame[
                            "displacement"
                        ],
                        "observation.tactile_right.distributed_force": right_tactile_frame[
                            "distributed_force"
                        ],
                        "observation.tactile_right.wrench": right_tactile_frame["wrench"],
                        "observation.tactile.timestamp": np.array(
                            [
                                left_tactile_frame["timestamp"],
                                right_tactile_frame["timestamp"],
                            ],
                            dtype=np.float64,
                        ),
                        "action": action.astype(np.float32),
                        "obj_init": obj_init,
                    },
                    task=args.task,
                )
                frames_in_episode += 1

            preview.show(base_image, wrist_image)
    finally:
        preview.close()
        teleop.close()
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()


if __name__ == "__main__":
    main()
