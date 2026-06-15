import mujoco
model = mujoco.MjModel.from_xml_path("./urdf/ur5_rg2.urdf")
print("Model loaded successfully!")