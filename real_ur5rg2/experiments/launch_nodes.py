import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.tactile.tac3d import DEFAULT_TAC3D_SDK_PATH, Tac3DDriver, Tac3DFrameCache
from gello.zmq_core.camera_node import ZMQServerCamera
from gello.zmq_core.robot_node import ZMQServerRobot
from gello.zmq_core.tactile_node import ZMQServerTactile


@dataclass
class Args:
    robot: str = "ur"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.102"
    enable_camera: bool = True
    enable_wrist_camera: bool = True
    enable_base_camera: bool = True
    wrist_camera_device_id: str = "412622271117"
    # base_camera_device_id: str = "939622073079"
    base_camera_device_id: str = "335522073597"
    wrist_camera_flip: bool = False
    base_camera_flip: bool = False
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    enable_depth: bool = False
    camera_backend: str = "lerobot"
    enable_tactile: bool = True
    left_tactile_port: int = 5100
    right_tactile_port: int = 5101
    left_tactile_device_id: str = "0"
    right_tactile_device_id: str = "1"
    tactile_sdk_path: str = DEFAULT_TAC3D_SDK_PATH
    tactile_module_name: str = "PyTac3D"
    tac3d_udp_port: int = 9988
    tac3d_max_queue_size: int = 5
    tactile_max_points: int = 400
    tactile_read_timeout_s: float = 10.0


def make_robot(args: Args):
    if args.robot == "ur":
        from gello.robots.ur import URRobot

        return URRobot(robot_ip=args.robot_ip, no_gripper=False)
    if args.robot == "bimanual_ur":
        from gello.robots.ur import URRobot

        left = URRobot(robot_ip="192.168.2.10", no_gripper=False)
        right = URRobot(robot_ip="192.168.1.10", no_gripper=False)
        return BimanualRobot(left, right)
    if args.robot in ("none", "print"):
        return PrintRobot(7)
    raise NotImplementedError("Choose one of: ur, bimanual_ur, none, print")


def launch_robot_server(args: Args) -> None:
    robot = make_robot(args)
    robot_server = ZMQServerRobot(robot, port=args.robot_port, host=args.hostname)
    print(f"Starting robot server on port {args.robot_port}")

    threads = [threading.Thread(target=robot_server.serve, daemon=True)]

    def make_camera(device_id: str, flip: bool):
        if args.camera_backend == "lerobot":
            from gello.cameras.lerobot_realsense_camera import LeRobotRealSenseCamera

            return LeRobotRealSenseCamera(
                device_id=device_id,
                flip=flip,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            )
        if args.camera_backend == "gello":
            from gello.cameras.realsense_camera import RealSenseCamera

            return RealSenseCamera(
                device_id=device_id,
                flip=flip,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                enable_depth=args.enable_depth,
            )
        raise ValueError("camera_backend must be 'lerobot' or 'gello'")

    if args.enable_camera and args.enable_wrist_camera:
        wrist_camera = make_camera(
            device_id=args.wrist_camera_device_id,
            flip=args.wrist_camera_flip,
        )
        wrist_camera_server = ZMQServerCamera(
            camera=wrist_camera,
            port=args.wrist_camera_port,
            host=args.hostname,
        )
        print(f"Starting wrist camera server on port {args.wrist_camera_port}")
        threads.append(threading.Thread(target=wrist_camera_server.serve, daemon=True))

    if args.enable_camera and args.enable_base_camera:
        base_camera = make_camera(
            device_id=args.base_camera_device_id,
            flip=args.base_camera_flip,
        )
        base_camera_server = ZMQServerCamera(
            camera=base_camera,
            port=args.base_camera_port,
            host=args.hostname,
        )
        print(f"Starting base camera server on port {args.base_camera_port}")
        threads.append(threading.Thread(target=base_camera_server.serve, daemon=True))

    def make_tactile(frame_cache: Tac3DFrameCache, device_id: str) -> Tac3DDriver:
        return Tac3DDriver(
            frame_cache=frame_cache,
            device_id=device_id,
            max_points=args.tactile_max_points,
            read_timeout_s=args.tactile_read_timeout_s,
        )

    if args.enable_tactile:
        tactile_cache = Tac3DFrameCache(
            sdk_path=args.tactile_sdk_path,
            module_name=args.tactile_module_name,
            udp_port=args.tac3d_udp_port,
            max_queue_size=args.tac3d_max_queue_size,
        )
        print(f"Listening for Tac3D Desktop UDP frames on port {args.tac3d_udp_port}")

        left_tactile = make_tactile(tactile_cache, args.left_tactile_device_id)
        left_tactile_server = ZMQServerTactile(
            tactile=left_tactile,
            port=args.left_tactile_port,
            host=args.hostname,
        )
        print(f"Starting left Tac3D tactile server on port {args.left_tactile_port}")
        threads.append(threading.Thread(target=left_tactile_server.serve, daemon=True))

        right_tactile = make_tactile(tactile_cache, args.right_tactile_device_id)
        right_tactile_server = ZMQServerTactile(
            tactile=right_tactile,
            port=args.right_tactile_port,
            host=args.hostname,
        )
        print(f"Starting right Tac3D tactile server on port {args.right_tactile_port}")
        threads.append(threading.Thread(target=right_tactile_server.serve, daemon=True))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main(args: Args) -> None:
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
