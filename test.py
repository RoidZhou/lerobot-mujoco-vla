from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
import cv2
import numpy as np

# 相机配置，暂时在这里写配置
config1 = RealSenseCameraConfig(
    serial_number_or_name="335522073597",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=False,
    rotation=Cv2Rotation.NO_ROTATION
)

config2 = RealSenseCameraConfig(
    serial_number_or_name="412622271117",
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

# 创建窗口
cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)

while True:
    # 读取图像
    rgb_agent = camera1.read()
    rgb_ego = camera2.read()
    
    # 转换为BGR格式用于显示
    vis_image_bgr = cv2.cvtColor(rgb_agent, cv2.COLOR_RGB2BGR)
    vis_wrist_image_bgr = cv2.cvtColor(rgb_ego, cv2.COLOR_RGB2BGR)
    
    # 调整大小（可选，根据需要）
    vis_image_bgr = cv2.resize(vis_image_bgr, (640, 480))
    vis_wrist_image_bgr = cv2.resize(vis_wrist_image_bgr, (640, 480))
    
    # 水平拼接显示
    vis = np.hstack([vis_image_bgr, vis_wrist_image_bgr])
    
    # 实时显示
    cv2.imshow('Camera View', vis)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
camera1.disconnect()
camera2.disconnect()
cv2.destroyAllWindows()