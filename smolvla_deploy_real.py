from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision
import cv2
import time
device = 'cuda'

try:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='/home/mel/VLA/lerobot-mujoco-tutorial/bc_data_4')
except:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size= 2, n_action_steps=2, num_steps=50)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# We can now instantiate our policy with this config and the dataset stats.
policy = SmolVLAPolicy.from_pretrained('./ckpt/smolvla_omy/checkpoints/last/pretrained_model',  config=cfg, dataset_stats=dataset_metadata.stats)
# You can load the trained policy from hub if you don't have the resources to train it.
# policy = SmolVLAPolicy.from_pretrained("Jeongeun/omy_pnp_pi0", config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)


from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation

# 相机配置，暂时在这里写配置
config1 = RealSenseCameraConfig(
    serial_number_or_name="939622073079",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=False,
    rotation=Cv2Rotation.NO_ROTATION
)

config2 = RealSenseCameraConfig(
    serial_number_or_name="335522073597",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=False,
    rotation=Cv2Rotation.NO_ROTATION
)

# 相机初始化
camera1 = RealSenseCamera(config1)
camera2 = RealSenseCamera(config2)

camera1.connect()
camera2.connect()

# 机器人初始化，端口号和ip地址要改成真实的
from gello.robots.ur import URRobot
URrobot = URRobot(robot_ip="192.168.1.102", no_gripper=False)

# 环境初始化，传入相机和机器人实例
from mujoco_env.y_env2_realsensor import SimpleEnv2
xml_path = './asset/example_scene_y_copy.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle', camera1=camera1, camera2=camera2, URrobot=URrobot)

def resize_rgb(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

from torchvision import transforms
# Approach 1: Using torchvision.transforms
def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
     Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0], shape C×H×W
    ])

try:
    period = 1.0 / 20.0
    step = 0
    PnPEnv.reset(seed=0)
    policy.reset()
    policy.eval()
    save_image = True
    IMG_TRANSFORM = get_default_transform()
    while PnPEnv.env.is_viewer_alive():
        start = time.time()
        PnPEnv.step_env() #发送控制cmd命令，更新环境状态
        if True:
            # Get the current state of the environment
            # 这里也改成真实的
            state = PnPEnv.get_joint_state()[:6] # 获得机器人的状态
            
            # Get the current image from the environment
            image, wrist_image = PnPEnv.grab_image()

            # 如果 image 已经是 numpy 数组 (H, W, C)
            # image = resize_rgb(image)
            # image = torch.from_numpy(image).float()/255
            # image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            image = IMG_TRANSFORM(image)

            # wrist_image = resize_rgb(wrist_image)
            # wrist_image = torch.from_numpy(wrist_image).float()/255
            # wrist_image = wrist_image.permute(2, 0, 1).unsqueeze(0).to(device)
            wrist_image = Image.fromarray(wrist_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = IMG_TRANSFORM(wrist_image)
            print("instruction :", PnPEnv.instruction)
            # 这里的数据用的image，wrist_image，state都是真实的
            data = {
                'observation.state': torch.tensor([state]).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                'task': [PnPEnv.instruction],
            }
            end0 = time.time()
            print(f"time 0 : {end0 - start}")
            # Select an action
            action = policy.select_action(data)
            action = action[0,:7].cpu().detach().numpy()
            # Take a step in the environment
            _ = PnPEnv.step(action)
            step += 1
            end = time.time()
            print(f"time : {end - start}")
            elapsed = time.time() - start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

finally:
    # PnPEnv.env.close_viewer()
    cv2.destroyAllWindows()
    camera1.disconnect()
    camera2.disconnect()


