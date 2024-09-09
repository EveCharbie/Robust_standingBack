"""
Created on Thu Aug 29 14:31:08 2024

@author: anais
"""

import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt
from graph_simu import graph_all_comparaison, get_created_data_from_pickle, time_to_percentage

# Solution with and without holonomic constraints
path_sol = "/home/mickaelbegon/Documents/Anais/Results_simu"
sol_holo = path_sol + "/" + "Salto_close_loop_landing_5phases_V80.pkl"
sol2 = path_sol + "/" + "Salto_5phases_V11.pkl"
path_model = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/Model/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)

data = pd.read_pickle(sol_holo)
data = get_created_data_from_pickle(sol_holo)
data2 = pd.read_pickle(sol2)

# Preparation data
time_end_phase = []
for i in range(len(data["time"])):
    time_end_phase.append(data["time"][i][-1])
time_end_phase_pourcentage = time_to_percentage(np.vstack(time_end_phase))

time_end_phase2 = []
for i in range(len(data2["time"])):
    time_end_phase2.append(data2["time"][i][-1])
time_end_phase2_pourcentage = time_to_percentage(np.vstack(time_end_phase2))
dof_names_tau = ["Shoulder", "Elbow",
                 "Hip", "Knee", "Ankle"]

time = np.vstack(data["time"])
time_pourcentage = time_to_percentage(time)
time2 = np.vstack(data2["time"])  # data2["time_all"]
time_pourcentage2 = time_to_percentage(time2)

# Difference time
time_diff = data["time"][-1][-1] - data2["time"][-1][-1]

# Duree phase solutions with and without holonomic constraints
time_phase_sol_holo = []
time_phase_sol2 = []

for i in range(len(data["time"])):
    time_phase_sol_holo.append(data["time"][i][-1] - data["time"][i][1])
    time_phase_sol2.append(data2["time"][i][-1] - data2["time"][i][1])

time_diff_phase = [a - b for a, b in zip(time_phase_sol_holo, time_phase_sol2)]

# Diminution des torques
# Par rapport entre premiere et dernier valeur
# Coude
diff_tau_elbow_sol_holo = (data["tau"][2][1][1] - data["tau"][2][1][-1]) / data["tau"][2][1][1] * 100
diff_tau_elbow_sol2 = (data2["tau"][2][1][1] - data2["tau"][2][1][-1]) / data2["tau"][2][1][1] * 100

# Hanche
diff_tau_hip_sol_holo = (data["tau"][2][2][1] - data["tau"][2][2][-1]) / data["tau"][2][2][1] * 100
diff_tau_hip_sol2 = (data2["tau"][2][2][1] - data2["tau"][2][2][-1]) / data2["tau"][2][2][1] * 100

# Par moyenne
# Coude
tau_mean_elbow_sol_holo = np.mean(data["tau"][2][1])
tau_std_elbow_sol_holo = np.std(data["tau"][2][1])
tau_mean_elbow_sol2 = np.mean(data2["tau"][2][1])
tau_std_elbow_sol2 = np.std(data2["tau"][2][1])

# Hanche
tau_mean_hip_sol_holo = np.mean(data["tau"][2][2])
tau_std_hip_sol_holo = np.std(data["tau"][2][2])
tau_mean_hip_sol2 = np.mean(data2["tau"][2][2])
tau_std_hip_sol2 = np.std(data2["tau"][2][2])

# Graphique
#graph_all_comparaison(sol_holo, sol2)

# Inertie
inertie_sol_holo = np.zeros((data["q_all"].shape[1], 3))
inertie_sol2 = np.zeros((data2["q_all"].shape[1], 3))
for i in range(data["q_all"].shape[1]):
    inertie_sol_holo[i, :] = np.diagonal(model.bodyInertia(data["q_all"][:, i]).to_array()).T
    inertie_sol2[i, :] = np.diagonal(model.bodyInertia(data2["q_all"][:, i]).to_array()).T

inertia_total_holo = np.sqrt(inertie_sol_holo[:, 0] ** 2 + inertie_sol_holo[:, 1] ** 2 + inertie_sol_holo[:, 2] ** 2)
inertia_total_sol2 = np.sqrt(inertie_sol2[:, 0] ** 2 + inertie_sol2[:, 1] ** 2 + inertie_sol2[:, 2] ** 2)

inertia_holo = np.concatenate((inertie_sol_holo, inertia_total_holo[:, np.newaxis]), axis=1)
inertia_2 = np.concatenate((inertie_sol2, inertia_total_sol2[:, np.newaxis]), axis=1)

# Calcul energie moyenne par phase et moyenne totale

# table_inertie_sol2 = np.zeros((4, len(data2["time"])+1))
# table_inertie_holo = np.zeros((4, len(data["time"])+1))
# for j in range(0,4):
#    frame_start_sol2 = 0
#    frame_end_sol2 = 0
#    frame_start_holo = 0
#    frame_end_holo = 0
#    for i in range(len(data["time"])+1):
#        if i < len(data["time"]):
#            frame_end_sol2 = frame_end_sol2 + data2["time"][i].shape[0]
#            table_inertie_sol2[j,i] = np.mean(inertia_2[frame_start_sol2:frame_end_sol2, j])
#            frame_start_sol2 = frame_start_sol2 + data2["time"][i].shape[0]
#            frame_end_holo = frame_end_holo + data["time"][i].shape[0]
#            table_inertie_holo[j,i] = np.mean(inertia_holo[frame_start_holo:frame_end_holo, j])
#            frame_start_holo = frame_start_holo + data["time"][i].shape[0]
#        else:
#            table_inertie_sol2[j,i] = np.mean(inertia_2[:, j])
#            table_inertie_holo[j,i] = np.mean(inertia_holo[:, j])


axs_names = ["x", "y", "z", "total"]
num_col = 0
num_line = 0

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3)

y_max = np.max([abs(inertia_2[:, 0:3]), abs(inertia_holo[:, 0:3])])

axs = []
for i in range(3):
    axs.append(fig.add_subplot(gs[0, i]))
axs.append(fig.add_subplot(gs[1, :2]))

for nb_ax in range(len(axs_names)):
    if nb_ax < 3:
        axs[nb_ax].plot(time_pourcentage2, inertia_2[:, nb_ax], color="tab:blue", label="without \nconstraints",
                        alpha=0.75, linewidth=1)
        axs[nb_ax].plot(time_pourcentage, inertia_holo[:, nb_ax], color="tab:orange",
                        label="with holonomics \nconstraints", alpha=0.75, linewidth=1)

        for xline in range(len(time_end_phase)):
            axs[nb_ax].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--", linewidth=0.7)
            axs[nb_ax].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--", linewidth=0.7)
    else:
        axs[nb_ax].plot(time_pourcentage2, inertia_2[:, nb_ax], color="tab:blue", label="without \nconstraints",
                        alpha=0.75, linewidth=1)
        axs[nb_ax].plot(time_pourcentage, inertia_holo[:, nb_ax], color="tab:orange",
                        label="with holonomics \nconstraints", alpha=0.75, linewidth=1)

        for xline in range(len(time_end_phase)):
            axs[nb_ax].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--", linewidth=0.7)
            axs[nb_ax].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--", linewidth=0.7)

    if nb_ax < 3:
        axs[nb_ax].set_ylim(-y_max + (-y_max * 0.1), y_max + (y_max * 0.1))
    axs[nb_ax].set_title(axs_names[nb_ax], fontsize=8)
    axs[nb_ax].grid(True, linewidth=0.4)
    axs[nb_ax].tick_params(axis='both', which='major', labelsize=6)
    handles, labels = axs[0].get_legend_handles_labels()
    axs_legend = fig.add_subplot(gs[1, 2])  # Create an empty subplot for the legend
    axs_legend.legend(handles, labels, loc='center', fontsize=8)
    axs_legend.axis('off')
    axs[0].set_ylabel("Inertia [kg/m2]", fontsize=7)
    axs[3].set_ylabel("Inertia [kg/m2]", fontsize=7)
    axs[3].set_xlabel('Time [%]', fontsize=7)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.show()

    fig.savefig("Inertia.svg", format="svg")

# Energy expenditure (intégrale de la somme de la valeur absolue de tau multiplier par la vitesse angulaire le tout multiplier par dt)
    # Sol Holo
time = np.vstack([array[:-1,:] for array in data["time"]])
intervalle_temps = time[1:] - time[:-1]
intervalle_temps = intervalle_temps[intervalle_temps!=0]
tau = np.hstack(data["tau"])
qdot = np.hstack([array[:,:-1] for array in data["qdot"]])

time_integral = time.flatten()
energy_holo = np.trapz(np.abs(tau*qdot[3:, :]), time_integral)
energy_holo_all = np.abs(tau[:, :]*qdot[3:, :])
energy_holo_total = energy_holo_all.sum(axis=0)

    # Sol 2
time2 = np.vstack([array[:-1,:] for array in data2["time"]])
tau2 = np.hstack(data2["tau"])
qdot2 = np.hstack([array[:,:-1] for array in data2["qdot"]])

time2_integral = time2.flatten()
energy_sol2 = np.trapz(np.abs(tau2*qdot2[3:, :]), time2_integral)
energy_sol2_all = np.abs(tau2[:, :]*qdot2[3:, :])
energy_sol2_total = energy_sol2_all.sum(axis=0)

# TODO: Corriger
#table_detail_energy_sol2 = np.zeros((5, len(data2["time"])+1))
#table_detail_energy_holo = np.zeros((5, len(data["time"])+1))
#for j in range(0,5):
#    frame_start_sol2 = 0
#    frame_end_sol2 = 0
#    frame_start_holo = 0
#    frame_end_holo = 0
#    for i in range(len(data["time"])+1):
#        if i < len(data["time"]):
#            frame_end_sol2 = frame_end_sol2 + data2["time"][i].shape[0]
#            table_detail_energy_sol2[j,i] = np.trapz(np.abs(tau2*qdot2[3:, frame_start_sol2:frame_end_sol2]), time2_integral[frame_start_sol2:frame_end_sol2])
#            frame_start_sol2 = frame_start_sol2 + data2["time"][i].shape[0]
#            frame_end_holo = frame_end_holo + data["time"][i].shape[0]
#            table_detail_energy_holo[j,i] = np.trapz(np.abs(tau*qdot[3:, frame_start_holo:frame_end_holo]), time_integral[frame_start_holo:frame_end_holo])
#            frame_start_holo = frame_start_holo + data["time"][i].shape[0]
#        else:
#            table_detail_energy_sol2[j,i] = np.sum(table_detail_energy_sol2)
#            table_detail_energy_holo[j,i] = np.sum(table_detail_energy_holo)


#Diff energy
energy_diff = energy_holo - energy_sol2

# Preparation data

dof_names_tau = ["Shoulder", "Elbow",
                 "Hip", "Knee", "Ankle"]

# Figure Energy expenditure
fig, axs = plt.subplots(2, 3)
num_col = 1
num_line = 0

y_max_1 = np.max([abs(energy_sol2_all[0:2]), abs(energy_holo_all[0:2])])
y_max_2 = np.max([abs(energy_sol2_all[2:]), abs(energy_holo_all[2:])])

axs[0, 0].plot([], [], color="tab:orange", label="with holonomics \nconstraints")
axs[0, 0].plot([], [], color="tab:blue", label="without \nconstraints")
axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
axs[0, 0].axis('off')

for nb_seg in range(energy_holo_all.shape[0]):
    axs[num_line, num_col].plot(time_pourcentage_tau2, energy_sol2_all[nb_seg], color="tab:blue", alpha=0.75,
                                linewidth=1, label="without \nconstraints")
    axs[num_line, num_col].plot(time_pourcentage_tau, energy_holo_all[nb_seg], color="tab:orange", alpha=0.75,
                                linewidth=1, label="with holonomics \nconstraints")
    for xline in range(len(time_end_phase)):
        axs[num_line, num_col].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                                       linewidth=0.7)
        axs[num_line, num_col].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                                       linewidth=0.7)
    axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
    axs[num_line, num_col].set_xlim(0, 100)
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
fig.savefig("Power.svg", format="svg")

# Power total
plt.plot(time_pourcentage_tau2, energy_sol2_total, color="tab:blue", alpha=0.75,
         linewidth=1, label="without \nconstraints")
plt.plot(time_pourcentage_tau, energy_holo_total, color="tab:orange", alpha=0.75,
         linewidth=1, label="with holonomics \nconstraints")

# Ajouter des lignes verticales
for xline in range(len(time_end_phase)):
    plt.axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                linewidth=0.7)
    plt.axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                linewidth=0.7)

plt.xlabel('Time [%]', fontsize=8)
plt.ylabel("Power [W]", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(True, linewidth=0.4)
plt.xlim(0, 100)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout()
fig.savefig("C:/Users/anais/Desktop/Cours/3A/Prélab/Recherche_stage/Stage_Montreal/Image/Power_total.svg", format="svg")


# Calcul energie moyenne par phase et moyenne totale

table_energy_sol2 = np.zeros((5, len(data2["time"])+1))
table_energy_holo = np.zeros((5, len(data["time"])+1))
for j in range(0,5):
    frame_start_sol2 = 0
    frame_end_sol2 = 0
    frame_start_holo = 0
    frame_end_holo = 0
    for i in range(len(data["time"])+1):
        if i < len(data["time"]):
            frame_end_sol2 = frame_end_sol2 + data2["time"][i].shape[0]
            table_energy_sol2[j,i] = np.mean(energy_sol2_all[j, frame_start_sol2:frame_end_sol2])
            frame_start_sol2 = frame_start_sol2 + data2["time"][i].shape[0]
            frame_end_holo = frame_end_holo + data["time"][i].shape[0]
            table_energy_holo[j,i] = np.mean(energy_holo_all[j, frame_start_holo:frame_end_holo])
            frame_start_holo = frame_start_holo + data["time"][i].shape[0]
        else:
            table_energy_sol2[j,i] = np.mean(energy_sol2_all[j, :])
            table_energy_holo[j,i] = np.mean(energy_holo_all[j, :])

    #Diff energy
energy_diff = energy_holo - energy_sol2

# Figure tau
fig, axs = plt.subplots(2, 3)
num_col = 1
num_line = 0

y_max_1 = np.max([abs(energy_sol2_all[0:2]), abs(energy_holo_all[0:2])])
y_max_2 = np.max([abs(energy_sol2_all[2:]), abs(energy_holo_all[2:])])

axs[0, 0].plot([], [], color="tab:orange", label="with holonomics \nconstraints")
axs[0, 0].plot([], [], color="tab:blue", label="without \nconstraints")
axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
axs[0, 0].axis('off')

for nb_seg in range(energy_holo_all.shape[0]):
    axs[num_line, num_col].step(range(len(energy_sol2_all[nb_seg])), energy_sol2_all[nb_seg], color="tab:blue", alpha=0.75,
                                linewidth=1, label="without \nconstraints", where='mid')
    axs[num_line, num_col].step(range(len(energy_holo_all[nb_seg])), energy_holo_all[nb_seg], color="tab:orange", alpha=0.75,
                                linewidth=1, label="with holonomics \nconstraints", where='mid')
    for xline in range(len(time_end_phase)):
        axs[num_line, num_col].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                                       linewidth=0.7)
        axs[num_line, num_col].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                                       linewidth=0.7)
    axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
    axs[num_line, num_col].set_xlim(0, 100)
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
axs[0, 1].set_ylabel("Energy expenditure [...]", fontsize=7)  # Arm Rotation
axs[1, 0].set_ylabel("Energy expenditure [...]", fontsize=7)  # Leg Rotation
axs[0, 2].set_yticklabels([])
axs[1, 1].set_yticklabels([])
axs[1, 2].set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)
fig.savefig("Energy_expenditure.svg", format="svg")

fig = plt.figure()
plt.plot(time, tau[0,:]*qdot[4, :], color='r', label=["Holo"])
plt.plot(time2,tau2[0,:]*qdot2[4, :], color='g', label=["Sol 2"])
plt.ylabel("Energy expenditure")
plt.xlabel("Time [s]")
plt.legend()
plt.show()