import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("model/scene.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model=model,data=data)