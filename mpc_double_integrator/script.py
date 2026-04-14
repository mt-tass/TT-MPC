import mujoco
import mujoco.viewer
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("model/scene.xml")
data = mujoco.MjData(model)
time_step = model.opt.timestep
N = 45
Q = np.diag([1000.0,10.0])
R = np.array([[0.1]])
x_destination = np.array([1.0,0.0])

A_d = np.array([[1,time_step],[0,1]])
B_d = np.array([[0],[time_step]]) 

skip = 5
u_last = 0.0
frame_count = 0

history = {"time": [], "pos": [], "vel": [], "ctrl": []}

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        start = time.time()
        history["time"].append(data.time)
        history["pos"].append(data.qpos[0])
        history["vel"].append(data.qvel[0])
        history["ctrl"].append(u_last)
        if data.qpos[0] >= 1.0:
            break

        if frame_count%skip == 0 :
            x_current = np.array([data.qpos[0] ,data.qvel[0]])
            x_predic = cp.Variable((N+1,2)) 
            u_predic = cp.Variable((N,1)) 
            cost = 0
            constraints = [x_predic[0] == x_current]
            for num in range(N):
                cost += cp.quad_form(x_predic[num]-x_destination,Q)
                cost += cp.quad_form(u_predic[num],R)
                constraints += [x_predic[num+1] == x_predic[num] @ A_d.T + u_predic[num] @ B_d.T]
                constraints += [cp.abs(u_predic[num]) <= 10]
            cost += cp.quad_form(x_predic[N]-x_destination,Q)
            problem = cp.Problem(cp.Minimize(cost),constraints=constraints)
            problem.solve(solver=cp.OSQP , warm_start=True , eps_abs=1e-3, eps_rel=1e-3)

            if u_predic.value is not None:
                u_last = float(u_predic.value[0][0])
        
        data.ctrl[0] = u_last
        mujoco.mj_step(model,data)
        viewer.sync()
        frame_count += 1
        
        diff = time.time() - start
        if time_step > diff:
            time.sleep(time_step - diff)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
axs[0].plot(history["time"], history["pos"], label="Actual Position", color='blue')
axs[0].axhline(y=x_destination[0], color='red', linestyle='--', label="Target")
axs[0].set_ylabel("Position (m)")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(history["time"], history["vel"], label="Velocity", color='green')
axs[1].set_ylabel("Velocity (m/s)")
axs[1].grid(True)

axs[2].step(history["time"], history["ctrl"], label="Control Effort (Force)", color='purple')
axs[2].axhline(y=10, color='black', linestyle=':', alpha=0.5)
axs[2].axhline(y=-10, color='black', linestyle=':', alpha=0.5, label="Constraints")
axs[2].set_ylabel("Control Input")
axs[2].set_xlabel("Time (s)")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
