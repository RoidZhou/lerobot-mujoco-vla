# Real UR5 + RG2 Keyboard LeRobot Collection

This folder adapts `ur5_rg2_collect_data_smolvla.py` from MuJoCo to a real UR5 + RG2 setup.

It uses keyboard teleoperation, not GELLO leader-arm teleoperation.

## 0. Install real-robot dependencies

`rtde_control` and `rtde_receive` are import module names. The pip package name is `ur_rtde`:

```bash
pip install ur_rtde
```

If you use RealSense cameras from this launcher, also make sure `pyrealsense2`, `pyzmq`, `pygame`, and `opencv-python` are installed in the same conda environment.

## 1. Start robot and camera nodes

Run from the project root:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur
```

Defaults:

- robot ZMQ: `127.0.0.1:6001`
- wrist RealSense: `127.0.0.1:5000`, device `335522073597`
- base RealSense: `127.0.0.1:5001`, device `939622073079`
- UR IP: `192.168.1.102`

You can override these with `--robot-ip`, `--wrist-camera-device-id`, and `--base-camera-device-id`.

The launcher defaults to RGB-only camera streams at 15 FPS to reduce dual-camera USB bandwidth. If one camera repeatedly reports frame timeouts, check that its serial number matches the connected device and try another USB3 port. You can reduce bandwidth further:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur --camera-width 424 --camera-height 240 --camera-fps 15
```

Only enable depth when you really need it:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur --enable-depth
```

## 2. Collect LeRobot data with keyboard

In a second terminal:

```bash
python real_ur5rg2/collect_lerobot_smolvla_real.py --task "Insert the bolt into the nut." --num-episodes 20
```

If OpenCV was installed without GUI support, disable the camera preview:

```bash
python real_ur5rg2/collect_lerobot_smolvla_real.py --task "Insert the bolt into the nut." --num-episodes 20 --no-preview
```

Keyboard controls match the simulation teleop style:

- `W/S/A/D`: move TCP in the x/y plane
- `R/F`: move TCP up/down
- Arrow keys and `Q/E`: rotate TCP
- `T/G`: screw down/up motion
- `SPACE`: toggle RG2 open/close
- `Z`: clear the current episode buffer
- `ENTER`: save the current episode
- `ESC`: exit, saving the current episode if it has frames

Like `ur5_rg2_collect_data_smolvla.py`, recording starts when a motion or gripper command is issued. Press `ENTER` to mark the episode done and save it.

Dataset schema:

- `observation.image`: base camera RGB image, `256x256x3`
- `observation.wrist_image`: wrist camera RGB image, `256x256x3`
- `observation.state`: real UR joint state, first 6 joints
- `action`: real UR5 + RG2 joint/gripper state, 7 dimensions
- `obj_init`: zero placeholder, 9 dimensions, kept for compatibility with existing training code

Use `--overwrite` to recreate the dataset root, or `--resume` to append to an existing complete LeRobot dataset.
If the dataset path already exists, the script will also ask interactively whether to overwrite, continue saving, or quit.
