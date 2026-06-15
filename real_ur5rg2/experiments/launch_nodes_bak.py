import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.cameras.realsense_camera import RealSenseCamera
from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.camera_node import ZMQServerCamera
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "ur"
    robot_port: int = 6001
    wrist_camera_port: int = 5002
    base_camera_port: int = 5003
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

    if args.enable_camera and args.enable_wrist_camera:
        wrist_camera = RealSenseCamera(
            device_id=args.wrist_camera_device_id,
            flip=args.wrist_camera_flip,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            enable_depth=args.enable_depth,
        )
        wrist_camera_server = ZMQServerCamera(
            camera=wrist_camera,
            port=args.wrist_camera_port,
            host=args.hostname,
        )
        print(f"Starting wrist camera server on port {args.wrist_camera_port}")
        threads.append(threading.Thread(target=wrist_camera_server.serve, daemon=True))

    if args.enable_camera and args.enable_base_camera:
        base_camera = RealSenseCamera(
            device_id=args.base_camera_device_id,
            flip=args.base_camera_flip,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            enable_depth=args.enable_depth,
        )
        base_camera_server = ZMQServerCamera(
            camera=base_camera,
            port=args.base_camera_port,
            host=args.hostname,
        )
        print(f"Starting base camera server on port {args.base_camera_port}")
        threads.append(threading.Thread(target=base_camera_server.serve, daemon=True))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main(args: Args) -> None:
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
