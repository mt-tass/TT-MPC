import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('robotstudio_so101/scene.xml')
data = mujoco.MjData(model)

mujoco.mj_resetData(model,data)
# print(model.nq)
# print(model.njnt)
# print(data.qpos[3])
# print(data.qvel[3])

# for i in range(model.njnt):
#     print(model.jnt_range[i][0])
#     print(model.jnt_range[i][1])

frames = int(10.0/model.opt.timestep)
with mujoco.viewer.launch_passive(model,data) as viewer:
    mujoco.mj_resetData(model,data)
    for frame in range(frames):
        for i in range(model.njnt):
            min = model.jnt_range[i][0]
            max = model.jnt_range[i][1]
            target = min + (max-min)*(frame/frames)
            error = target - data.qpos[i]
            d_error = -data.qvel[i]
            data.ctrl[i] = 10*error + d_error
        mujoco.mj_step(model,data)
        viewer.sync()

# for i in range(model.njnt):
#     mujoco.mj_resetData(model,data)
#     min = model.jnt_range[i][0]
#     max = model.jnt_range[i][1]
#     with mujoco.viewer.launch_passive(model,data) as viewer :
#         for frame in range(frames):
#             target = min + (max-min)*(frame/frames)
#             error = target-data.qpos[i]
#             d_error = -data.qvel[i]
#             data.ctrl[i] = 10*error + 1*d_error
#             mujoco.mj_step(model,data)
#             viewer.sync()

