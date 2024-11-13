"""
Created on Thu Aug 29 14:31:08 2024

@author: anais
"""
import os
import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt

from data_analysis.graph_multi_start import num_col
from graph_simu import graph_all_comparaison # , get_created_data_from_pickle, time_to_percentage

# Solution with and without holonomic constraints
path_sol = "/home/mickaelbegon/Documents/Anais/Results_simu"
pathwithout = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/holonomic_research/solutions/Salto_5phases_VEve_final/"
# pathCL = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/holonomic_research/solutions_CL/Salto_close_loop_landing_5phases_VEve_final/"
sol_CL = path_sol + "/" + "Salto_close_loop_landing_5phases_VEve12.pkl"
# sol_without = path_sol + "/" + "Salto_5phases_VEve14.pkl"
path_model = "../Model/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)

min_cost_without = np.inf
for file in os.listdir(pathwithout):
    if file.endswith("CVG.pkl"):
        data = pd.read_pickle(pathwithout + file)
        if data["cost"] < min_cost_without:
            min_cost_without = data["cost"]
            sol_without = pathwithout + file
print("Min cost without: ", min_cost_without)

PLOT_TAU_FLAG = False
PLOT_INERTIA_FLAG = True
PLOT_ENERY_FLAG = True
format_graph = "svg"
phase_delimiter = ["-", "--", ":", "-.", "-"]

data_CL = pd.read_pickle(sol_CL)
data_without = pd.read_pickle(sol_without)

# Preparation data_CL
time_end_phase_CL = []
for i in range(len(data_CL["time"])):
    time_end_phase_CL.append(data_CL["time"][i][-1])
# time_end_phase_pourcentage_CL = time_to_percentage(np.vstack(time_end_phase_CL))

time_end_phase_without = []
for i in range(len(data_without["time"])):
    time_end_phase_without.append(data_without["time"][i][-1])
# time_end_phase_pourcentage_without = time_to_percentage(np.vstack(time_end_phase_without))
dof_names_tau = ["Shoulder", "Elbow",
                 "Hip", "Knee", "Ankle"]

time_CL = data_CL["time"]
# time_pourcentage_CL = time_to_percentage(time_CL)
time_vector_CL = np.vstack(data_CL["time"]).reshape(-1, ) - time_CL[0][-1]
time_end_phase_CL = np.array(time_end_phase_CL) - time_CL[0][-1]

time_without = data_without["time"]
# time_pourcentage_without = time_to_percentage(time_without)
time_vector_without = np.vstack(data_without["time"]).reshape(-1, ) - time_without[0][-1]
time_end_phase_without = np.array(time_end_phase_without) - time_without[0][-1]

time_max_graph = max(time_vector_without[-1], time_vector_CL[-1])
time_min_graph = min(time_vector_without[0], time_vector_CL[0])

# Duree phase solutions with and without holonomic constraints
time_phaseCL = []
time_phasewithout = []

for i in range(len(data_CL["time"])):
    time_phaseCL.append(data_CL["time"][i][-1] - data_CL["time"][i][0])
    time_phasewithout.append(data_without["time"][i][-1] - data_without["time"][i][0])

print("*** Phase_time *** \nCL :", time_phaseCL, "\nwithout : ", time_phasewithout)
print("Total CL :", np.sum(time_phaseCL), "\nTotal without : ", np.sum(time_phasewithout))

# Graphique
if PLOT_TAU_FLAG:
    graph_all_comparaison(data_CL, data_without, format_graph)


if PLOT_INERTIA_FLAG:
    # Inertia
    inertia_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    inertia_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        inertia_CL[i, :] = np.diagonal(model.bodyInertia(data_CL["q_all"][:, i]).to_array()).T
        inertia_without[i, :] = np.diagonal(model.bodyInertia(data_without["q_all"][:, i]).to_array()).T
        
    fig, ax = plt.subplots(4, 1, figsize=(8, 9))
    ax[0].plot(time_vector_without, inertia_without[:, 0], color="tab:blue", label="Kinematic tucking constraints", alpha=0.75, linewidth=1)#time_pourcentage_without.flatten(),
    ax[0].plot(time_vector_CL, inertia_CL[:, 0], color="tab:orange", label="Holonomic tucking constraints", alpha=0.75, linewidth=1)#time_pourcentage_CL,
    for xline in range(len(time_end_phase_CL) - 1):
        ax[0].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline], linewidth=0.7)
        ax[0].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline], linewidth=0.7)
    ax[0].set_ylabel("Moment of inertia\n" + r"[$kg.m^2$]")
    ax[0].set_xlim(time_min_graph, time_max_graph)
    ax[0].grid(True, linewidth=0.4)
    ax[0].legend(bbox_to_anchor=(0.95, 1.45), ncol=2)


    # Angular momentum
    ang_mom_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    ang_mom_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        ang_mom_CL[i, :] = model.angularMomentum(data_CL["q_all"][:, i], data_CL["qdot_all"][:, i]).to_array()
        ang_mom_without[i, :] = model.angularMomentum(data_without["q_all"][:, i], data_without["qdot_all"][:, i]).to_array()

    ax[1].plot(time_vector_without, ang_mom_without[:, 0], color="tab:blue", label="Kinematic tucking constraints",
               alpha=0.75, linewidth=1)
    ax[1].plot(time_vector_CL, ang_mom_CL[:, 0], color="tab:orange", label="Holonomic tucking constraints",
               alpha=0.75, linewidth=1)
    for xline in range(len(time_end_phase_CL) - 1):
        ax[1].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline], linewidth=0.7)
        ax[1].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline], linewidth=0.7)
    ax[1].set_ylabel("Angular momentum\n" + r"[$kg.m^2/s$]")
    ax[1].set_xlim(time_min_graph, time_max_graph)
    ax[1].grid(True, linewidth=0.4)


    # Body velocity
    body_velo_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    body_velo_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        body_velo_CL[i, :] = model.bodyAngularVelocity(data_CL["q_all"][:, i], data_CL["qdot_all"][:, i]).to_array()
        body_velo_without[i, :] = model.bodyAngularVelocity(data_without["q_all"][:, i], data_without["qdot_all"][:, i]).to_array()

    ax[2].plot(time_vector_without, body_velo_without[:, 0], color="tab:blue", label="Kinematic tucking constraints",
               alpha=0.75, linewidth=1)
    ax[2].plot(time_vector_CL, body_velo_CL[:, 0], color="tab:orange", label="Holonomic tucking constraints",
               alpha=0.75, linewidth=1)
    for xline in range(len(time_end_phase_CL) - 1):
        ax[2].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline], linewidth=0.7)
        ax[2].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline], linewidth=0.7)
    ax[2].set_ylabel("Somersault velocity\n" + r"[$rad/s$]")
    ax[2].set_xlim(time_min_graph, time_max_graph)
    ax[2].grid(True, linewidth=0.4)


    # Centrifugal effect
    centricugal_CL = np.sqrt(model.mass()) * ang_mom_CL[:, 0]**2 / (inertia_CL[:, 0] ** (3/2))
    centricugal_without = np.sqrt(model.mass()) * ang_mom_without[:, 0]**2 / (inertia_without[:, 0] ** (3/2))

    ax[3].plot(time_vector_without, centricugal_without, color="tab:blue", label="Kinematic tucking constraints",
               alpha=0.75, linewidth=1)
    ax[3].plot(time_vector_CL, centricugal_CL, color="tab:orange", label="Holonomic tucking constraints",
               alpha=0.75, linewidth=1)
    for xline in range(len(time_end_phase_CL) - 1):
        ax[3].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline], linewidth=0.7)
        ax[3].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline], linewidth=0.7)
    ax[3].set_ylabel("Centrifugal effect\n" + r"[$N$]")
    ax[3].set_xlim(time_min_graph, time_max_graph)
    ax[3].grid(True, linewidth=0.4)
    ax[3].set_xlabel("Time [s]")


    fig.subplots_adjust()
    plt.savefig("Inertia" + "." + format_graph, format = format_graph)
    #plt.show()

# Energy expenditure (intégrale de la somme de la valeur absolue de tau multiplier par la vitesse angulaire le tout multiplier par dt)
time_CL = np.vstack([array[:-1,:] for array in data_CL["time"]])
intervalle_temps_CL = time_CL[1:] - time_CL[:-1]
intervalle_temps_CL = intervalle_temps_CL[intervalle_temps_CL!=0]
tau_CL = np.hstack(data_CL["tau"])
qdot_CL = np.hstack([array[:,:-1] for array in data_CL["qdot"]])

energy_CL = np.trapz(np.abs(tau_CL*qdot_CL[3:, :]), time_CL.T)
energy_CL_all = np.abs(tau_CL[:, :]*qdot_CL[3:, :])
energy_CL_total = energy_CL_all.sum(axis=0)

time_without = np.vstack([array[:-1,:] for array in data_without["time"]])
tau_without = np.hstack(data_without["tau"])
qdot_without = np.hstack([array[:,:-1] for array in data_without["qdot"]])

energy_without = np.trapz(np.abs(tau_without*qdot_without[3:, :]), time_without.T)
energy_without_all = np.abs(tau_without[:, :]*qdot_without[3:, :])
energy_without_total = energy_without_all.sum(axis=0)

#Diff energy
print("Energy expanditure CL: ", energy_CL_total)
print("Energy expanditure without: ", energy_without_total)
print("Energy expanditure CL (tucked phase) : ", energy_CL[2])
print("Energy expanditure without (tucked phase) : ", energy_without[2])
print("Energy difference : ", (energy_without_total - energy_CL_total) / energy_CL_total * 100, " %")


# Figure Energy expenditure
time_tau_CL = np.vstack((time_CL[:20],
                                    time_CL[21:21+20],
                                    time_CL[21+21:21+21+30],
                                    time_CL[21+21+31:21+21+31+30],
                                    time_CL[21+21+31+31:21+21+31+31+30]))
time_tau_without = np.vstack((time_without[:20],
                                        time_without[21:21+20],
                                        time_without[21+21:21+21+30],
                                        time_without[21+21+31:21+21+31+30],
                                        time_without[21+21+31+31:21+21+31+31+30]))
# time_pourcentage_tau_CL = np.vstack((time_pourcentage_CL[:20],
#                                     time_pourcentage_CL[21:21+20],
#                                     time_pourcentage_CL[21+21:21+21+30],
#                                     time_pourcentage_CL[21+21+31:21+21+31+30],
#                                     time_pourcentage_CL[21+21+31+31:21+21+31+31+30]))
# time_pourcentage_tau_without = np.vstack((time_pourcentage_without[:20],
#                                         time_pourcentage_without[21:21+20],
#                                         time_pourcentage_without[21+21:21+21+30],
#                                         time_pourcentage_without[21+21+31:21+21+31+30],
#                                         time_pourcentage_without[21+21+31+31:21+21+31+31+30]))

fig, axs = plt.subplots(2, 3)
num_col = 1
num_line = 0

y_max_1 = np.max([abs(energy_without_all[0:2]), abs(energy_CL_all[0:2])])
y_max_2 = np.max([abs(energy_without_all[2:]), abs(energy_CL_all[2:])])

axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking constraints")
axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
axs[0, 0].axis('off')

for nb_seg in range(energy_CL_all.shape[0]):
    axs[num_line, num_col].plot(time_tau_without, energy_without_all[nb_seg], color="tab:blue", alpha=0.75,
                                linewidth=1, label="Kinematic tucking constraints")
    axs[num_line, num_col].plot(time_tau_CL, energy_CL_all[nb_seg], color="tab:orange", alpha=0.75,
                                linewidth=1, label="Holonomic tucking constraints")
    for xline in range(len(time_end_phase_CL) - 1):
        axs[num_line, num_col].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline],
                                       linewidth=0.7)
        axs[num_line, num_col].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline],
                                       linewidth=0.7)
    axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
    axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
    axs[num_line, num_col].grid(True, linewidth=0.4)

    # Réduire la taille des labels des xticks et yticks
    axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
    if num_line == 0:
        axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
    else:
        axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

    num_col += 1
    if num_col == 3:
        num_col = 0
        num_line += 1
    if num_line == 1:
        axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

# Y_label
axs[0, 1].set_ylabel("Power [W]", fontsize=7)  # Arm Rotation
axs[1, 0].set_ylabel("Power [W]", fontsize=7)  # Leg Rotation
axs[0, 2].set_yticklabels([])
axs[1, 1].set_yticklabels([])
axs[1, 2].set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig("Power"+ "." + format_graph, format=format_graph)

# Power total
plt.plot(time_tau_without, energy_without_total, color="tab:blue", alpha=0.75,
         linewidth=1, label="Kinematic tucking constraints")
plt.plot(time_tau_CL, energy_CL_total, color="tab:orange", alpha=0.75,
         linewidth=1, label="Holonomic tucking constraints")

# Ajouter des lignes verticales
for xline in range(len(time_end_phase_CL) - 1):
    plt.axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline],
                linewidth=0.7)
    plt.axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline],
                linewidth=0.7)

plt.xlabel('Time [%]', fontsize=8)
plt.ylabel("Power [W]", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(True, linewidth=0.4)
plt.xlim(time_min_graph, time_max_graph)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout()
fig.savefig("Power_total"+ "." + format_graph, format=format_graph)


# Calcul energie moyenne par phase et moyenne totale

table_energy_without = np.zeros((5, len(data_without["time"])+1))
table_energy_CL = np.zeros((5, len(data_CL["time"])+1))
for j in range(0,5):
    frame_startwithout = 0
    frame_endwithout = 0
    frame_start_CL = 0
    frame_end_CL = 0
    for i in range(len(data_CL["time"])+1):
        if i < len(data_CL["time"]):
            frame_endwithout = frame_endwithout + data_without["time"][i].shape[0]
            table_energy_without[j, i] = np.mean(energy_without_all[j, frame_startwithout:frame_endwithout])
            frame_startwithout = frame_startwithout + data_without["time"][i].shape[0]
            frame_end_CL = frame_end_CL + data_CL["time"][i].shape[0]
            table_energy_CL[j, i] = np.mean(energy_CL_all[j, frame_start_CL:frame_end_CL])
            frame_start_CL = frame_start_CL + data_CL["time"][i].shape[0]
        else:
            table_energy_without[j, i] = np.mean(energy_without_all[j, :])
            table_energy_CL[j, i] = np.mean(energy_CL_all[j, :])

#Diff energy
energy_diff = energy_CL - energy_without

# Figure tau
if PLOT_ENERY_FLAG:
    fig, axs = plt.subplots(2, 3)
    num_col = 1
    num_line = 0

    y_max_1 = np.max([abs(energy_without_all[0:2]), abs(energy_CL_all[0:2])])
    y_max_2 = np.max([abs(energy_without_all[2:]), abs(energy_CL_all[2:])])

    axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking constraints")
    axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
    axs[0, 0].axis('off')

    for nb_seg in range(energy_CL_all.shape[0]):
        axs[num_line, num_col].step(range(len(energy_without_all[nb_seg])), energy_without_all[nb_seg], color="tab:blue", alpha=0.75,
                                    linewidth=1, label="Kinematic tucking constraints", where='mid')
        axs[num_line, num_col].step(range(len(energy_CL_all[nb_seg])), energy_CL_all[nb_seg], color="tab:orange", alpha=0.75,
                                    linewidth=1, label="Holonomic tucking constraints", where='mid')
        for xline in range(len(time_end_phase_CL) - 1):
            axs[num_line, num_col].axvline(time_end_phase_CL[xline], color="tab:orange", linestyle=phase_delimiter[xline],
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_without[xline], color="tab:blue", linestyle=phase_delimiter[xline],
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)

        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Energy expenditure", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Energy expenditure", fontsize=7)  # Leg Rotation
    axs[0, 2].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 2].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig("Energy_expenditure"+ "." + format_graph, format=format_graph)

    fig = plt.figure()
    plt.plot(time_CL, tau_CL[0, :]*qdot_CL[4, :], color='r', label="Holonomic tucking constraints")
    plt.plot(time_without, tau_without[0,:]*qdot_without[4, :], color='g', label="Kinematic tucking constraints")
    plt.ylabel("Energy expenditure")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.savefig("Energy"+ "." + format_graph, format=format_graph)
    #plt.show()
