"""
Created on Thu Aug 29 14:31:08 2024

@author: anais
"""
import os
import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt

from graph_simu import graph_all_comparaison, plot_vertical_time_lines

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

PLOT_TAU_FLAG = True
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

    plot_vertical_time_lines(time_end_phase_CL[0], time_end_phase_without[0], ax[0], color="k",
                             linestyle='-', linewidth=0.5)
    plot_vertical_time_lines(time_end_phase_CL[1], time_end_phase_without[1], ax[0], color=None,
                             linestyle=phase_delimiter[1], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[2], time_end_phase_without[2], ax[0], color=None,
                             linestyle=phase_delimiter[2], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[3], time_end_phase_without[3], ax[0], color=None,
                             linestyle=phase_delimiter[3], linewidth=None)

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

    plot_vertical_time_lines(time_end_phase_CL[0], time_end_phase_without[0], ax[1], color="k",
                             linestyle='-', linewidth=0.5)
    plot_vertical_time_lines(time_end_phase_CL[1], time_end_phase_without[1], ax[1], color=None,
                             linestyle=phase_delimiter[1], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[2], time_end_phase_without[2], ax[1], color=None,
                             linestyle=phase_delimiter[2], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[3], time_end_phase_without[3], ax[1], color=None,
                             linestyle=phase_delimiter[3], linewidth=None)

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

    plot_vertical_time_lines(time_end_phase_CL[0], time_end_phase_without[0], ax[2], color="k",
                             linestyle='-', linewidth=0.5)
    plot_vertical_time_lines(time_end_phase_CL[1], time_end_phase_without[1], ax[2], color=None,
                             linestyle=phase_delimiter[1], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[2], time_end_phase_without[2], ax[2], color=None,
                             linestyle=phase_delimiter[2], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[3], time_end_phase_without[3], ax[2], color=None,
                             linestyle=phase_delimiter[3], linewidth=None)

    ax[2].set_ylabel("Somersault velocity\n" + r"[$rad/s$]")
    ax[2].set_xlim(time_min_graph, time_max_graph)
    ax[2].grid(True, linewidth=0.4)


    # Centrifugal effect
    centricugal_CL = model.mass() * body_velo_CL[:, 0]**2 * np.sqrt(inertia_CL[:, 0] / model.mass())
    centricugal_without = model.mass() * body_velo_without[:, 0]**2 * np.sqrt(inertia_without[:, 0] / model.mass())

    ax[3].plot(time_vector_without, centricugal_without, color="tab:blue", label="Kinematic tucking constraints",
               alpha=0.75, linewidth=1)
    ax[3].plot(time_vector_CL, centricugal_CL, color="tab:orange", label="Holonomic tucking constraints",
               alpha=0.75, linewidth=1)

    plot_vertical_time_lines(time_end_phase_CL[0], time_end_phase_without[0], ax[3], color="k",
                             linestyle='-', linewidth=0.5)
    plot_vertical_time_lines(time_end_phase_CL[1], time_end_phase_without[1], ax[3], color=None,
                             linestyle=phase_delimiter[1], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[2], time_end_phase_without[2], ax[3], color=None,
                             linestyle=phase_delimiter[2], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[3], time_end_phase_without[3], ax[3], color=None,
                             linestyle=phase_delimiter[3], linewidth=None)

    ax[3].set_ylabel("Centrifugal pseudo-force\n" + r"[$N$]")
    ax[3].set_xlim(time_min_graph, time_max_graph)
    ax[3].grid(True, linewidth=0.4)
    ax[3].set_xlabel("Time [s]")


    fig.subplots_adjust()
    plt.savefig("Inertia" + "." + format_graph, format = format_graph)
    #plt.show()

"""
# Energy expenditure (intégrale de la somme de la valeur absolue de tau multiplier par la vitesse angulaire le tout multiplier par dt)
time_CL = np.vstack([array[:-1,:] for array in data_CL["time"]])
intervalle_temps_CL = time_CL[1:] - time_CL[:-1]
intervalle_temps_CL = intervalle_temps_CL[intervalle_temps_CL!=0]
tau_CL = np.hstack(data_CL["tau"])
qdot_CL = np.hstack([array[:,:-1] for array in data_CL["qdot"]])

energy_CL = np.trapz(np.abs(tau_CL*qdot_CL[3:, :]), time_CL.T)
energy_CL_all = np.abs(tau_CL[:, :]*qdot_CL[3:, :])
energy_CL_total = np.sum(energy_CL_all)

time_without = np.vstack([array[:-1,:] for array in data_without["time"]])
tau_without = np.hstack(data_without["tau"])
qdot_without = np.hstack([array[:,:-1] for array in data_without["qdot"]])

energy_without = np.trapz(np.abs(tau_without*qdot_without[3:, :]), time_without.T)
energy_without_all = np.abs(tau_without[:, :]*qdot_without[3:, :])
energy_without_total = np.sum(energy_without_all)

#Diff energy
print("Energy expanditure CL: ", energy_CL_total)
print("Energy expanditure without: ", energy_without_total)
print("Energy expanditure CL (tucked phase) : ", energy_CL[2])
print("Energy expanditure without (tucked phase) : ", energy_without[2])
print("Energy difference : ", (energy_CL_total - energy_without_total) / energy_CL_total * 100, " %")
"""

if PLOT_ENERY_FLAG:
    tau_CL = np.hstack(data_CL["tau"])
    tau_without = np.hstack(data_without["tau"])

    qdot_CL = np.hstack([array[:, :-1] for array in data_CL["qdot"]])
    qdot_without = np.hstack([array[:, :-1] for array in data_without["qdot"]])

    time_tau_CL = np.hstack((time_vector_CL[:20],
                             time_vector_CL[21:21 + 20],
                             time_vector_CL[21 + 21:21 + 21 + 30],
                             time_vector_CL[21 + 21 + 31:21 + 21 + 31 + 30],
                             time_vector_CL[21 + 21 + 31 + 31:21 + 21 + 31 + 31 + 30]))
    time_tau_without = np.hstack((time_vector_without[:20],
                                  time_vector_without[21:21 + 20],
                                  time_vector_without[21 + 21:21 + 21 + 30],
                                  time_vector_without[21 + 21 + 31:21 + 21 + 31 + 30],
                                  time_vector_without[21 + 21 + 31 + 31:21 + 21 + 31 + 31 + 30]))

    power_without = np.abs(tau_without*qdot_without[3:, :])
    power_CL = np.abs(tau_CL*qdot_CL[3:, :])
    power_total_without = np.sum(power_without, axis=0)
    power_total_CL = np.sum(power_CL, axis=0)
    energy_without = np.sum(np.trapz(power_without, time_tau_without))
    energy_CL = np.sum(np.trapz(power_CL, time_tau_CL))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(time_tau_CL, power_total_CL, color="tab:orange", label="Holonomic tucking constraints")
    axs[0].plot(time_tau_without, power_total_without, color="tab:blue", label="Kinematic tucking constraints")

    plot_vertical_time_lines(time_end_phase_CL[0], time_end_phase_without[0], axs[0], color="k",
                             linestyle='-', linewidth=0.5)
    plot_vertical_time_lines(time_end_phase_CL[1], time_end_phase_without[1], axs[0], color=None,
                             linestyle=phase_delimiter[1], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[2], time_end_phase_without[2], axs[0], color=None,
                             linestyle=phase_delimiter[2], linewidth=None)
    plot_vertical_time_lines(time_end_phase_CL[3], time_end_phase_without[3], axs[0], color=None,
                             linestyle=phase_delimiter[3], linewidth=None)

    axs[1].bar([0, 1], [energy_CL, energy_without], color=["tab:orange", "tab:blue"])

    axs[0].legend(bbox_to_anchor=(2.0, 1.3), ncol=2)
    axs[0].grid(True, linewidth=0.4)
    axs[0].set_ylabel("Joint power \n[J/s]")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])


    axs[1].set_ylabel("Energy expenditure \n[J]")
    axs[1].set_xticks([0, 1], ["HTC", "KTC"])

    plt.subplots_adjust(wspace=0.4, bottom=0.2, top=0.8)
    plt.savefig("Energy"+ "." + format_graph, format=format_graph)
    #plt.show()
