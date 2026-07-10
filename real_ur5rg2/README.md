# Real UR5 + RG2 Keyboard LeRobot Collection

This folder adapts `ur5_rg2_collect_data_smolvla.py` from MuJoCo to a real UR5 + RG2 setup.

It uses keyboard teleoperation, not GELLO leader-arm teleoperation.

## 0. Install real-robot dependencies

`rtde_control` and `rtde_receive` are import module names. The pip package name is `ur_rtde`:

```bash
pip install ur_rtde
```

If you use RealSense cameras from this launcher, also make sure `pyrealsense2`, `pyzmq`, `pygame`, and `opencv-python` are installed in the same conda environment.

Tac3D's Python SDK also imports `ruamel.yaml` and `cv2`:

```bash
pip install ruamel.yaml opencv-python
```

For Tac3D tactile collection, make sure the Tac3D SDK Python path is available. The launcher defaults to:

```bash
/media/mel/D64E-50BC/Tac3D-SDK-v3.3.0-20250407/Tac3D-SDK-v3.3.0/Tac3D-API/python/PyTac3D
```

If the external drive is mounted under a different name, the launcher also scans `/media/mel/*/Tac3D-SDK-v3.3.0-20250407/.../PyTac3D`. You can always override the path explicitly:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur \
  --tactile-sdk-path /media/mel/62F7-892C/Tac3D-SDK-v3.3.0-20250407/Tac3D-SDK-v3.3.0/Tac3D-API/python/PyTac3D
```

Before starting `launch_nodes.py`, start the Tac3D core senders in separate terminals:

```bash
cd /home/mel/ybzhou/lerobot-mujoco-tutorial/local_tac3d_core
./Tac3D -c config/DL1-GWM0013 -i 127.0.0.1 -p 9988
```

```bash
cd /home/mel/ybzhou/lerobot-mujoco-tutorial/local_tac3d_core
./Tac3D -c config/DL1-GWM0018 -i 127.0.0.1 -p 9988
```

`config/DL1-GWM0013` streams SN `DL1-GWM0013`, and `config/DL1-GWM0018` streams SN `DL1-GWM0018`. If one Tac3D process cannot open its camera, check `inputSrc` in that config's `sensor.yaml`; with two USB Tac3D cameras, one config may need `inputSrc: 1`.

Then start robot/camera/tactile ZMQ with explicit left/right Tac3D SNs:

```bash
python real_ur5rg2/experiments/launch_nodes.py \
  --robot ur \
  --wrist-camera-device-id 412622271117 \
  --base-camera-device-id 335522073597 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --left-tactile-device-id DL1-GWM0013 \
  --right-tactile-device-id DL1-GWM0018

python real_ur5rg2/collect_lerobot_smolvla_real.py \
  --task "Insert the bolt into the nut." \
  --num-episodes 20 \
  --no-preview \
  --no-collect-force \
  --collect-tactile
```
```

To test the pipeline with one sensor only, point both logical fingers at the same received sensor.

If you get `Permission denied` from the SDK on the `/media/...` drive, copy `Tac3D-Core/linux-x86_64` to a Linux filesystem such as this repository's ignored `local_tac3d_core/` directory and run it from there.

For tactile-only visualization, you can then run `PyTac3D_Displayer2.py`. For data collection, do not keep `PyTac3D_Displayer2.py` open on the same UDP port, because `launch_nodes.py` also binds `9988` to receive the Tac3D frames.

## 1. Start robot and camera nodes

After the Tac3D core sender is running, run from the project root:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur
```

Defaults:

- robot ZMQ: `127.0.0.1:6001`
- wrist RealSense: `127.0.0.1:5000`, device `335522073597`
- base RealSense: `127.0.0.1:5001`, device `939622073079`
- left Tac3D: `127.0.0.1:5100`, device/index `0`
- right Tac3D: `127.0.0.1:5101`, device/index `1`
- Tac3D Desktop UDP input: `9988`
- UR IP: `192.168.1.102`

You can override these with `--robot-ip`, `--wrist-camera-device-id`, `--base-camera-device-id`, `--left-tactile-device-id`, and `--right-tactile-device-id`.

`--left-tactile-device-id` and `--right-tactile-device-id` may be real Tac3D SN strings, or `0`/`1` to use the first/second sensor received on the Tac3D UDP stream. If Tac3D Desktop sends data to a different UDP port, pass it explicitly:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur \
  --tac3d-udp-port 9988
```

The tactile server waits up to 10 seconds for a matching Tac3D frame by default. If Tac3D Desktop takes longer to start streaming, increase both sides:

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur \
  --tactile-read-timeout-s 30

python real_ur5rg2/collect_lerobot_smolvla_real.py \
  --task "Insert the bolt into the nut." \
  --num-episodes 20 \
  --no-preview \
  --collect-force \
  --no-collect-tactile \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5
```

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

Tac3D remains the default tactile backend. To collect Paxini S2716_core tactile
data instead, switch `--tactile-source` and pass the USB serial port. One Paxini
sensor is stored as the left tactile stream and the right stream is zero-filled;
pass `--paxini-right-port` as well if you use two Paxini sensors.

```bash
python real_ur5rg2/collect_lerobot_smolvla_real.py \
  --task "Insert the bolt into the nut." \
  --num-episodes 20 \
  --no-preview \
  --collect-tactile \
  --tactile-source paxini \
  --paxini-left-port /dev/ttyUSB0 \
  --tactile-max-points 400
```

To reduce excessive x/y/z contact force and torque during bolt/nut collection, enable the
FTS-300-S force reader. The collector will tare the sensor, map the raw Modbus
axes like `ur5moverealrobot3.py` (`[Fx,Fy,Fz] -> [-Fx,-Fz,-Fy]` in tool frame),
map torque with the same reference script convention, and add a small 6D
Cartesian admittance retreat before sending each TCP target:

```bash
python real_ur5rg2/collect_lerobot_smolvla_real.py \
  --task "Insert the bolt into the nut." \
  --num-episodes 50 \
  --no-preview \
  --collect-force \
  --no-collect-tactile \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5
  --force-safety-threshold-n 5,5,5 \
  --force-safety-hard-stop-n 18,18,18 \
  --force-safety-max-correction-m 0.006 \
  --torque-safety-threshold-nm 0.5,0.5,0.5 \
  --torque-safety-hard-stop-nm 1.5,1.5,1.5 \
  --torque-safety-max-correction-rad 0.035
```
```bash
python real_ur5rg2/collect_lerobot_smolvla_real.py \
  --task "Insert the bolt into the nut." \
  --num-episodes 50 \
  --no-preview \
  --collect-force \
  --no-collect-tactile \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20

  python real_ur5rg2/collect_lerobot_smolvla_real.py   --task "Insert the bolt into the nut."   --num-episodes 50   --no-preview   --collect-force   --no-collect-tactile   --force-serial-port /dev/ttyUSB0   --force-serial-timeout-s 0.1   --force-serial-retries 5   --force-safety-threshold-n 10,10,20
```
If the robot retreats in the wrong direction on one axis, flip that axis with
`--force-safety-axis-signs`, for example `--force-safety-axis-signs -1,1,-1`.
Use `--no-force-safety` to record force/torque without applying the correction.

Keyboard controls match the simulation teleop style:

- `W/S/A/D`: move TCP in the x/y plane
- `R/F`: move TCP up/down
- Arrow keys and `Q/E`: rotate TCP
- `T/G`: toggle screw down/up motion; press the active key again to stop
- `SPACE`: toggle RG2 open/close
- `Z`: clear the current episode buffer
- `ENTER`: save the current episode
- `ESC`: exit, saving the current episode if it has frames

Like `ur5_rg2_collect_data_smolvla.py`, recording starts when a motion or gripper command is issued. Press `ENTER` to mark the episode done and save it.

Dataset schema:

- `observation.image`: base camera RGB image, `256x256x3`
- `observation.wrist_image`: wrist camera RGB image, `256x256x3`
- `observation.state`: real UR joint state, first 6 joints
- `observation.tactile_left.displacement`: left Tac3D 3D displacement field, `400x3` by default
- `observation.tactile_left.distributed_force`: left Tac3D 3D distributed force field, `400x3` by default
- `observation.tactile_left.wrench`: left Tac3D resultant force and torque, `[Fx, Fy, Fz, Tx, Ty, Tz]`
- `observation.tactile_right.displacement`: right Tac3D 3D displacement field, `400x3` by default
- `observation.tactile_right.distributed_force`: right Tac3D 3D distributed force field, `400x3` by default
- `observation.tactile_right.wrench`: right Tac3D resultant force and torque, `[Fx, Fy, Fz, Tx, Ty, Tz]`
- `observation.tactile.timestamp`: `[left_timestamp, right_timestamp]`
- `action`: real UR5 + RG2 joint/gripper state, 7 dimensions
- `obj_init`: zero placeholder, 9 dimensions, kept for compatibility with existing training code

Use `--tactile-max-points` on both commands if your Tac3D field has more or fewer taxels than 400. Frames with fewer points are zero-padded; frames with more points are truncated to keep the LeRobot schema fixed.

Use `--overwrite` to recreate the dataset root, or `--resume` to append to an existing complete LeRobot dataset.
If the dataset path already exists, the script will also ask interactively whether to overwrite, continue saving, or quit.


## dataset clean
```bash
python real_ur5rg2/merge_lerobot_datasets.py \
  --filter-root real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_merged \
  --drop-existing-episodes 0-19 \
  --in-place
```

## 3. infer without force
```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur \
  --tactile-read-timeout-s 30

python real_ur5rg2/infer_smolvla_real_ur5.py \
  --task "Insert the bolt into the nut." \
  --root /home/mel/ybzhou/lerobot-mujoco-tutorial/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_merged \
  --policy-path /home/mel/ybzhou/lerobot-mujoco-tutorial/ckpt/checkpoints_smolvla_boltnut_100_without_force/020000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5
```


## 34 infer with force

```bash
python real_ur5rg2/infer_smolvla_real_ur5_force_control.py \
  --device cuda:1 \
  --task "Insert the bolt into the nut." \
  --root /home/lab202/YBZHOU/lerobot-mujoco-vla/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_total \
  --policy-path /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/checkpoints_smolvla_force_boltnut_speedup_total/020000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20

# smolvla
## boltnut
'''
1. checkpoints_smolvla_wo_force_vqvae_boltnut_speedup_total200_4w         035000   /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae/latest.pt
2. checkpoints_pi0_force_vqvae_boltnut_speedup_total200_7_6_shareatten    035000   /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae_200/latest.pt
3. checkpoints_smolvla_wo_force_vqvae_boltnut_speedup_200_T-Rex_atten     030000   /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae_200/latest.pt
4. checkpoints_smolvla_force_position_wo_vq_wo_trex_w_forcepred_wfp02_v2  030000   


4 效果最好
'''

python real_ur5rg2/infer_smolvla_real_ur5_force_control.py \
  --device cuda:1 \
  --task "Insert the bolt into the nut." \
  --root /home/lab202/YBZHOU/lerobot-mujoco-vla/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_200 \
  --policy-path /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/checkpoints_smolvla_wo_force_vqvae_boltnut_speedup_total200_4w/035000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20 \
  --effort-key observation.force_torque \
  --force-vqvae-ckpt /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae/latest.pt

python real_ur5rg2/infer_smolvla_real_ur5_force_position_control.py \
  --device cuda:1 \
  --task "Insert the bolt into the nut." \
  --root /home/lab202/YBZHOU/lerobot-mujoco-vla/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_200 \
  --policy-path /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/checkpoints_smolvla_force_position_wo_vq_wo_trex_w_forcepred_wfp02_v3/025000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20 \
  --no-force-safety \
  --effort-key observation.force_torque \
  --force-vqvae-ckpt /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae_200/latest.pt \
  --no-force-position-require-contact \
  --force-position-contact-threshold-n 0.3 \
  --force-position-k-m-per-n 3e-3 \
  --force-position-max-down-step-m 0.0008 \
  --force-position-max-up-step-m 0.0002 \
  --force-position-direction-sign -1

## push button
'''
1. checkpoints_smolvla_wo_force_vqvae_pushbutton_100_T-Rex_atten2  025000 
2. checkpoints_smolvla_wo_force_vqvae_pushbutton_100_T-Rex_atten   035000 
3. checkpoints_smolvla_wo_force_vqvae_pushbutton_100_atten         020000

1 和 2 区别不大， 2 更稳一点， 都优于3，
'''
python real_ur5rg2/infer_smolvla_real_ur5_force_control.py \
  --device cuda:1 \
  --task "push-in socket button." \
  --root /home/lab202/YBZHOU/lerobot-mujoco-vla/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_pushbutton \
  --policy-path /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/checkpoints_smolvla_wo_force_vqvae_pushbutton_100_T-Rex_atten/035000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20 \
  --effort-key observation.force_torque \
  --force-vqvae-ckpt /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae_pb_100/latest.pt

# pi0
python real_ur5rg2/infer_smolvla_real_ur5_force_control.py \
  --device cuda:1 \
  --task "Insert the bolt into the nut." \
  --root /home/lab202/YBZHOU/lerobot-mujoco-vla/real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_200 \
  --policy-path /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/checkpoints_pi0_force_vqvae_boltnut_speedup_total200_7_5_v2/020000/pretrained_model \
  --collect-force \
  --force-serial-port /dev/ttyUSB0 \
  --force-serial-timeout-s 0.1 \
  --force-serial-retries 5 \
  --force-safety-threshold-n 10,10,20 \
  --effort-key observation.force_torque \
  --force-vqvae-ckpt /home/lab202/YBZHOU/lerobot-mujoco-vla/ckpt/force_vqvae_200/checkpoint_epoch002.pt
```
