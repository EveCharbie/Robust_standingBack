import os
import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt
from graph_simu import graph_all_comparaison, get_created_data_from_pickle, time_to_percentage

# Solution with and without holonomic constraints
path_sol = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/holonomic_research/solutions/Salto_5phases_VEve16"

path_sol_CL = "/home/mickaelbegon/Documents/Anais/Results_simu"
sol_CL = path_sol_CL + "/" + "Salto_close_loop_landing_5phases_VEve12.pkl"
data_CL = pd.read_pickle(sol_CL)

path_model = "../Model/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)


fig, axs = plt.subplots(2, 3, figsize=(10, 5))
axs[0, 0].plot([], [], color="k", label="with holonomics \nconstraints")
tau_CL = data_CL["tau_all"]

num_col = 1
num_line = 0
for nb_seg in range(tau_CL.shape[0]):
    axs[num_line, num_col].step(range(len(tau_CL[nb_seg])), tau_CL[nb_seg], color="k", alpha=0.75, linewidth=1,
                                label="with holonomics \nconstraints", where='mid')

    num_col += 1
    if num_col == 3:
        num_col = 0
        num_line += 1
    if num_line == 1:
        axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)


colors = ["tab:blue",
          "tab:orange",
          "tab:green",
          "tab:red",
          "tab:purple",
          "tab:brown",
          "tab:pink",
          "tab:gray",
          "tab:olive",
          "tab:cyan",
          "m"]
for file in os.listdir(path_sol):
    if file.endswith("CVG.pkl"):
        sol_without = path_sol + "/" + file
        data_without = pd.read_pickle(sol_without)
        i_trial = int(file.split("_")[1])

        tau_without = data_without["tau_all"]

        axs[0, 0].plot([], [], color=colors[i_trial], label="without \nconstraints")
        axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
        axs[0, 0].axis('off')

        num_col = 1
        num_line = 0
        for nb_seg in range(tau_CL.shape[0]):
            axs[num_line, num_col].plot(np.array([0, 100]), np.array([0, 0]), '-k', linewidth=0.5)
            axs[num_line, num_col].step(range(len(tau_without[nb_seg])), tau_without[nb_seg], color=colors[i_trial], alpha=0.75, linewidth=1, label=f"without \nconstraints {i_trial}", where='mid')

            axs[num_line, num_col].grid(True, linewidth=0.4)

            num_col += 1
            if num_col == 3:
                num_col = 0
                num_line += 1
            if num_line == 1:
                axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

        # Y_label
        axs[0, 1].set_ylabel("Joint torque [Nm]", fontsize=7)  # Arm Rotation
        axs[1, 0].set_ylabel("Joint torque [Nm]", fontsize=7)  # Leg Rotation

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig("tau_multi_start.png", format="png")
plt.show()