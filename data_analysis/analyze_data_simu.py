

import os
import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt

import sys
sys.path.append("../holonomic_research/")
from actuators import Joint, actuator_function
from actuator_constants import ACTUATORS

def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data_CL.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data_CL: All the data_CL of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp

def plot_vertical_time_lines(time_end_phase_CL, time_end_phase_without, ax, color, linestyle, linewidth):
    color_CL = "tab:orange" if color is None else color
    color_without = "tab:blue" if color is None else color
    linewidth = 0.7 if linewidth is None else linewidth
    ax.axvline(time_end_phase_CL, color=color_CL, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(time_end_phase_without, color=color_without, linestyle=linestyle, linewidth=linewidth)
    return


# Solution with and without holonomic constraints
path_sol = "/home/mickaelbegon/Documents/Anais/Results_simu"

path_without = "../holonomic_research/solutions/"
path_CL = "../holonomic_research/solutions_CL/"

path_model = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)

CONSIDER_ONLY_CONVERGED = True
if CONSIDER_ONLY_CONVERGED:
    end_file = "CVG.pkl"
else:
    end_file = ".pkl"

min_cost_without = np.inf
for file in os.listdir(path_without):
    if file.endswith(end_file):
        data = pd.read_pickle(path_without + file)
        if data["cost"] < min_cost_without:
            min_cost_without = data["cost"]
            sol_without = path_without + file
print("Min cost without: ", min_cost_without)

min_cost_CL = np.inf
for file in os.listdir(path_CL):
    if file.endswith(end_file):
        data = pd.read_pickle(path_CL + file)
        if data["cost"] < min_cost_CL:
            min_cost_CL = data["cost"]
            sol_CL = path_CL + file
print("Min cost CL: ", min_cost_CL)

data_CL = pd.read_pickle(sol_CL)
data_without = pd.read_pickle(sol_without)



PLOT_TAU_FLAG = True
PLOT_INERTIA_FLAG = True
PLOT_ENERY_FLAG = True
format_graph = "png"

n_shooting_plus_one = (21, 21, 31, 31, 31)
phase_delimiter = ["-", "--", ":", "-.", "-"]
dof_names = [
    "Pelvis \n(Translation Y)",
    "Pelvis \n(Translation Z)",
    "Pelvis",
    "Shoulder",
    "Elbow",
    "Hip",
    "Knee",
    "Ankle",
]
dof_names_tau = ["Shoulder", "Elbow", "Hip", "Knee", "Ankle"]

# Data to plot ----------------------------------------------

# Solution with closed-loop constraints
q_CL_rad = data_CL["q_all"][:, :]
q_CL_rad_original = data_CL["q_all"][:, :]
q_CL_rad[6, :] = q_CL_rad[6, :] * -1
q_CL_deg = np.vstack([q_CL_rad[0:2, :], q_CL_rad[2:, :] * 180 / np.pi])
qdot_CL_rad = data_CL["qdot_all"]
qdot_CL_rad[6, :] = qdot_CL_rad[6, :] * -1
qdot_CL_deg = np.vstack([qdot_CL_rad[0:2, :], qdot_CL_rad[2:, :] * 180 / np.pi])
tau_CL = data_CL["tau_all"]
taudot_CL = data_CL["taudot_all"] if "taudot_all" in data_CL.keys() else np.hstack(data_CL["taudot"])

time_CL = data_CL["time"]
time_end_phase_CL = []
time_end_phase_tau_CL = []
for i in range(len(time_CL)):
    time_end_phase_CL.append(time_CL[i][-1])
    time_end_phase_tau_CL.append(time_CL[i][-2])
time_vector_CL = (
        np.vstack(data_CL["time"]).reshape(
            -1,
        )
        - time_CL[0][-1]
)
time_end_phase_CL = np.array(time_end_phase_CL) - time_CL[0][-1]
time_control_CL = (
        np.vstack([arr[:-1, :] for arr in time_CL]).reshape(
            -1,
        )
        - time_CL[0][-1]
)

# Solution without closed-loop constraints
q_without_rad = data_without["q_all"][:, :]
q_without_rad_original = data_without["q_all"][:, :]
q_without_rad[6, :] = q_without_rad[6, :] * -1
q_without_deg = np.vstack([q_without_rad[0:2, :], q_without_rad[2:, :] * 180 / np.pi])
qdot_without_rad = data_without["qdot_all"]
qdot_without_rad[6, :] = qdot_without_rad[6, :] * -1
qdot_without_deg = np.vstack([qdot_without_rad[0:2, :], qdot_without_rad[2:, :] * 180 / np.pi])
tau_without = data_without["tau_all"]
taudot_without = data_without["taudot_all"] if "taudot_all" in data_without.keys() else np.hstack(
    data_without["taudot"])

time_without = data_without["time"]
time_end_phase_without = []
time_end_phase_tau_without = []
for i in range(len(time_without)):
    time_end_phase_without.append(time_without[i][-1])
    time_end_phase_tau_without.append(time_without[i][-2])
time_vector_without = (
        np.vstack(data_without["time"]).reshape(
            -1,
        )
        - time_without[0][-1]
)
time_end_phase_without = np.array(time_end_phase_without) - time_without[0][-1]
time_control_without = (
        np.vstack([arr[:-1, :] for arr in time_without]).reshape(
            -1,
        )
        - time_without[0][-1]
)

time_max_graph = max(time_vector_without[-1], time_vector_CL[-1])
time_min_graph = min(time_vector_without[0], time_vector_CL[0])

# Duree phase solutions with and without holonomic constraints
time_phase_CL = []
time_phase_without = []

for i in range(len(data_CL["time"])):
    time_phase_CL.append(data_CL["time"][i][-1] - data_CL["time"][i][0])
    time_phase_without.append(data_without["time"][i][-1] - data_without["time"][i][0])

print("*** Phase_time *** \nCL :", time_phase_CL, "\nwithout : ", time_phase_without)
print("Total CL :", np.sum(time_phase_CL), "\nTotal without : ", np.sum(time_phase_without))


plt.figure()
plt.plot(data_CL["lambda"][0, :], "b")
plt.plot(data_CL["lambda"][1, :], "r")
plt.savefig("lambda_tempo.png")
plt.show()


# Graphique
if PLOT_TAU_FLAG:

    # Min and max bounds on q
    min_bound_q = np.ones((q_without_rad.shape[0], time_vector_without.shape[0])) * -1000
    max_bound_q = np.ones((q_without_rad.shape[0], time_vector_without.shape[0])) * 1000
    for i_phase in range(5):
        idx_beginning = sum(n_shooting_plus_one[:i_phase])
        idx_end = sum(n_shooting_plus_one[: i_phase + 1]) - 1
        for i_dof in range(8):
            if data_without["min_bounds_q"][i_phase][i_dof, 0] > min_bound_q[i_dof, idx_beginning]:
                min_bound_q[i_dof, idx_beginning] = data_without["min_bounds_q"][i_phase][i_dof, 0]
            if data_without["max_bounds_q"][i_phase][i_dof, 0] < max_bound_q[i_dof, idx_beginning]:
                max_bound_q[i_dof, idx_beginning] = data_without["max_bounds_q"][i_phase][i_dof, 0]
        for i_node in range(idx_beginning + 1, idx_end):
            min_bound_q[:, i_node] = data_without["min_bounds_q"][i_phase][:, 1]
            max_bound_q[:, i_node] = data_without["max_bounds_q"][i_phase][:, 1]
        min_bound_q[:, idx_end] = data_without["min_bounds_q"][i_phase][:, 2]
        max_bound_q[:, idx_end] = data_without["max_bounds_q"][i_phase][:, 2]
    min_bound_q *= 180 / np.pi
    max_bound_q *= 180 / np.pi
    tempo_min_6 = min_bound_q[6, :].copy()
    min_bound_q[6, :] = -max_bound_q[6, :]
    max_bound_q[6, :] = -tempo_min_6

    # Figure q
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(q_CL_deg[0:2, :]), abs(q_without_deg[0:2, :])])
    y_max_2 = np.max([abs(q_CL_deg[2:5, :]), abs(q_without_deg[2:5, :])])
    y_max_3 = np.max([abs(q_CL_deg[5:, :]), abs(q_without_deg[5:, :])])
    for i_dof in range(q_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        plot_vertical_time_lines(
            time_end_phase_CL[0],
            time_end_phase_without[0],
            axs[num_line, num_col],
            color="k",
            linestyle="-",
            linewidth=0.5,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[1],
            time_end_phase_without[1],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[1],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[2],
            time_end_phase_without[2],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[2],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[3],
            time_end_phase_without[3],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[3],
            linewidth=None,
        )

        axs[num_line, num_col].fill_between(
            time_vector_without,
            max_bound_q[i_dof, :],
            np.ones(max_bound_q[i_dof, :].shape) * 1000,
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            np.ones(min_bound_q[i_dof, :].shape) * -1000,
            min_bound_q[i_dof, :],
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            max_bound_q[i_dof, :],
            np.ones(max_bound_q[i_dof, :].shape) * 1000,
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            np.ones(min_bound_q[i_dof, :].shape) * -1000,
            min_bound_q[i_dof, :],
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )

        axs[num_line, num_col].plot(
            time_vector_without,
            q_without_deg[i_dof, :],
            color="tab:blue",
            label="Kinematic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            q_CL_deg[i_dof, :],
            color="tab:orange",
            label="Holonomic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )

        axs[num_line, num_col].set_title(dof_names[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))

        num_col = num_col + 1
        if i_dof == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 0].set_ylabel("Position [m]", fontsize=7)  # Pelvis Translation
    axs[1, 0].set_ylabel(r"Joint angle [$^\circ$]", fontsize=7)  # Pelvis Rotation
    axs[2, 0].set_ylabel(r"Joint angle [$^\circ$]", fontsize=7)  # Thight Rotation

    # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
    axs[0, 0].fill_between(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        color="tab:blue",
        alpha=0.1,
        step="pre",
        linewidth=0.5,
        label="$q_{bounds}$ Kinematic tucking constraints",
    )
    axs[0, 0].fill_between(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        color="tab:orange",
        alpha=0.1,
        step="pre",
        linewidth=0.5,
        label="$q_{bounds}$ Holonomic tucking constraints",
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Ajouter la légende à la figure de la première ligne, troisième colonne
    axs[0, 2].legend(handles, labels, loc="center", fontsize=8)
    axs[0, 2].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("q" + "." + format_graph, format=format_graph)

    # Figure qdot
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(qdot_without_deg[0:2, :]), abs(qdot_without_deg[0:2, :])])
    y_max_2 = np.max([abs(qdot_without_deg[2:5, :]), abs(qdot_without_deg[2:5, :])])
    y_max_3 = np.max([abs(qdot_without_deg[5:, :]), abs(qdot_without_deg[5:, :])])
    for i_dof in range(qdot_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)
        axs[num_line, num_col].plot(
            time_vector_without,
            qdot_without_deg[i_dof],
            color="tab:blue",
            label="Kinematic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            qdot_CL_deg[i_dof],
            color="tab:orange",
            label="Holonomic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )

        plot_vertical_time_lines(
            time_end_phase_CL[0],
            time_end_phase_without[0],
            axs[num_line, num_col],
            color="k",
            linestyle="-",
            linewidth=0.5,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[1],
            time_end_phase_without[1],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[1],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[2],
            time_end_phase_without[2],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[2],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[3],
            time_end_phase_without[3],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[3],
            linewidth=None,
        )

        axs[num_line, num_col].set_title(dof_names[i_dof])

        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major")

        num_col = num_col + 1
        if i_dof == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

        # Y_label
        axs[0, 0].set_ylabel("Velocity [m/s]", fontsize=7)  # Pelvis Translation
        # axs[0, 1].set_yticklabels([])  # Pelvis Translation
        axs[1, 0].set_ylabel("F (+) / E (-) [" + r"$^\circ$/s" + "]", fontsize=7)  # Pelvis Rotation
        # axs[1, 1].set_yticklabels([])  # Arm Rotation
        # axs[1, 2].set_yticklabels([])  # Forearm Rotation
        axs[2, 0].set_ylabel("F (+) / E (-) [" + r"$^\circ$/s" + "]", fontsize=7)  # Thight Rotation
        # axs[2, 1].set_yticklabels([])  # Leg Rotation
        # axs[2, 2].set_yticklabels([])  # Foot Rotation
        # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Ajouter la légende à la figure de la première ligne, troisième colonne
        axs[0, 2].legend(handles, labels, loc="center", fontsize=7)
        axs[0, 2].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("qdot" + "." + format_graph, format=format_graph)


    tau_CL_min_bound = np.zeros((5, tau_CL.shape[1]))
    tau_CL_max_bound = np.zeros((5, tau_CL.shape[1]))
    tau_without_min_bound = np.zeros((5, tau_without.shape[1]))
    tau_without_max_bound = np.zeros((5, tau_without.shape[1]))
    for i_dof, key in enumerate(ACTUATORS.keys()):
        # Change the sign of the theta_opt for the knee
        if i_dof == 3:
            true_min_CL = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                -q_CL_rad[i_dof + 3],
            )
            true_max_CL = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                -q_CL_rad[i_dof + 3],
            )
            true_min_without = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                -q_without_rad[i_dof + 3],
            )
            true_max_without = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                -q_without_rad[i_dof + 3],
            )
            tau_CL_min_bound[i_dof, :] = -true_max_CL
            tau_CL_max_bound[i_dof, :] = -true_min_CL
            tau_without_min_bound[i_dof, :] = -true_max_without
            tau_without_max_bound[i_dof, :] = -true_min_without
        else:
            tau_CL_min_bound[i_dof, :] = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                q_CL_rad[i_dof + 3],
            )
            tau_CL_max_bound[i_dof, :] = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                q_CL_rad[i_dof + 3],
            )
            tau_without_min_bound[i_dof, :] = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                q_without_rad[i_dof + 3],
            )
            tau_without_max_bound[i_dof, :] = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                q_without_rad[i_dof + 3],
            )

    # Figure tau
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))
    num_col = 1
    num_line = 0

    y_max_1 = np.max([abs(tau_without[0:2, :]), abs(tau_CL[0:2, :])])
    y_max_2 = np.max([abs(tau_without[2:, :]), abs(tau_CL[2:, :])])

    axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking contraints")
    axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs[0, 0].fill_between(
        [], [], [], color="tab:orange", alpha=0.1, label=r"$\tilde{\tau}^{max}$ Holonomic tucking contraints", linewidth=0.5
    )
    axs[0, 0].fill_between(
        [], [], [], color="tab:blue", alpha=0.1, label=r"$\tilde{\tau}^{max}$ Kinematic tucking constraints", linewidth=0.5
    )
    axs[0, 0].legend(loc="center right", bbox_to_anchor=(0.9, 0.5), fontsize=8)
    axs[0, 0].axis("off")

    for i_dof in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)
        axs[num_line, num_col].fill_between(
            time_vector_without,
            tau_without_max_bound[i_dof],
            np.ones(tau_without_max_bound[i_dof].shape) * 1000,
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            np.ones(tau_without_max_bound[i_dof].shape) * -1000,
            tau_without_min_bound[i_dof],
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            tau_CL_max_bound[i_dof],
            np.ones(tau_CL_max_bound[i_dof].shape) * 1000,
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            np.ones(tau_CL_max_bound[i_dof].shape) * -1000,
            tau_CL_min_bound[i_dof],
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )

        axs[num_line, num_col].plot(
            time_vector_without,
            tau_without[i_dof],
            color="tab:blue",
            alpha=0.75,
            linewidth=1,
            label="Kinematic tucking constraints",
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            tau_CL[i_dof],
            color="tab:orange",
            alpha=0.75,
            linewidth=1,
            label="Holonomic tucking constraints",
        )

        plot_vertical_time_lines(
            time_end_phase_CL[0],
            time_end_phase_without[0],
            axs[num_line, num_col],
            color="k",
            linestyle="-",
            linewidth=0.5,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[1],
            time_end_phase_without[1],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[1],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[2],
            time_end_phase_without[2],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[2],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[3],
            time_end_phase_without[3],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[3],
            linewidth=None,
        )

        axs[num_line, num_col].set_title(dof_names_tau[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque [Nm]", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Joint torque [Nm]", fontsize=7)  # Leg Rotation
    # axs[0, 2].set_yticklabels([])
    # axs[1, 1].set_yticklabels([])
    # axs[1, 2].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("tau" + "." + format_graph, format=format_graph)

    # Tau ratio all phases
    tau_CL_ratio_all = np.zeros(tau_CL.shape)
    tau_without_ratio_all = np.zeros(tau_without.shape)

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    axs[0, 0].plot(
        time_vector_CL,
        np.sum(np.abs(tau_CL), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[0, 0].plot(
        time_vector_without,
        np.sum(np.abs(tau_without), axis=0),
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], axs[0, 0], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1],
        time_end_phase_without[1],
        axs[0, 0],
        color=None,
        linestyle=phase_delimiter[1],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2],
        time_end_phase_without[2],
        axs[0, 0],
        color=None,
        linestyle=phase_delimiter[2],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3],
        time_end_phase_without[3],
        axs[0, 0],
        color=None,
        linestyle=phase_delimiter[3],
        linewidth=None,
    )

    for i_dof in range(5):
        for i_node in range(tau_CL.shape[1]):
            if tau_CL[i_dof, i_node] > 0:
                tau_CL_ratio_all[i_dof, i_node] = tau_CL[i_dof, i_node] / tau_CL_max_bound[i_dof, i_node]
            else:
                tau_CL_ratio_all[i_dof, i_node] = np.abs(tau_CL[i_dof, i_node] / tau_CL_min_bound[i_dof, i_node])
            if tau_without[i_dof, i_node] > 0:
                tau_without_ratio_all[i_dof, i_node] = tau_without[i_dof, i_node] / tau_without_max_bound[i_dof, i_node]
            else:
                tau_without_ratio_all[i_dof, i_node] = np.abs(
                    tau_without[i_dof, i_node] / tau_without_min_bound[i_dof, i_node]
                )
        # axs[1, 0].step(time_tau_CL, tau_CL_ratio_all[i_dof, :], color="tab:orange",
        #               label="Holonomic tucking constraints", alpha=0.75, linewidth=1)
        # axs[1, 0].step(time_tau_without, tau_without_ratio_all[i_dof, :], color="tab:blue",
        #               label="Kinematic tucking constraints", alpha=0.75, linewidth=1)
    axs[1, 0].plot(
        time_vector_CL,
        np.sum(np.abs(tau_CL_ratio_all), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[1, 0].plot(
        time_vector_without,
        np.sum(np.abs(tau_without_ratio_all), axis=0),
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], axs[1, 0], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1],
        time_end_phase_without[1],
        axs[1, 0],
        color=None,
        linestyle=phase_delimiter[1],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2],
        time_end_phase_without[2],
        axs[1, 0],
        color=None,
        linestyle=phase_delimiter[2],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3],
        time_end_phase_without[3],
        axs[1, 0],
        color=None,
        linestyle=phase_delimiter[3],
        linewidth=None,
    )

    integral_tau_all_CL = np.trapz(np.sum(np.abs(tau_CL), axis=0), x=time_vector_CL)
    integral_tau_all_without = np.trapz(np.sum(np.abs(tau_without), axis=0), x=time_vector_without)
    axs[0, 1].bar([0, 1], [integral_tau_all_CL, integral_tau_all_without], color=["tab:orange", "tab:blue"])
    integral_tau_all_ratio_CL = np.trapz(np.sum(np.abs(tau_CL_ratio_all), axis=0), x=time_vector_CL)
    integral_tau_all_ratio_without = np.trapz(np.sum(np.abs(tau_without_ratio_all), axis=0), x=time_vector_without)
    axs[1, 1].bar([0, 1], [integral_tau_all_ratio_CL, integral_tau_all_ratio_without], color=["tab:orange", "tab:blue"])

    # axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Joint torque \n[Nm]")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Physiological joint torque ratio")
    axs[0, 1].set_xticks([0, 1], ["HTC", "KTC"])
    axs[1, 1].set_xticks([0, 1], ["HTC", "KTC"])
    axs[0, 1].set_ylabel(r"$\int{ | \tau | dt}$" + "\n[Nm.s]")
    axs[0, 0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])
    axs[1, 0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

    axs[1, 1].plot([], [], color="tab:orange", label="Holonomic tucking constraints")
    axs[1, 1].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    # axs[1, 1].fill_between([], [], [], color="tab:orange", label=r"$\int{| \tau / \tilde{\tau}^{max} | dt}$ HTC")
    # axs[1, 1].fill_between([], [], [], color="tab:blue", label=r"$\int{| \tau / \tilde{\tau}^{max} | dt}$ KTC")
    axs[1, 1].legend(loc="center right", bbox_to_anchor=(0.5, 3.1), ncol=2)

    axs[1, 1].set_ylabel(r"$\int{ | \tau/{\tilde{\tau^{max}}} | dt}$" + "\n[s]")
    plt.subplots_adjust(hspace=0.25, wspace=0.4, top=0.95)
    plt.savefig("tau_ratio_all" + "." + format_graph, format=format_graph)
    # plt.show()


    # Figure taudot
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))
    num_col = 1
    num_line = 0

    y_max_1 = np.max([abs(taudot_without[0:2, :]), abs(taudot_CL[0:2, :])])
    y_max_2 = np.max([abs(taudot_without[2:, :]), abs(taudot_CL[2:, :])])

    axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking contraints")
    axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs[0, 0].legend(loc="center right", bbox_to_anchor=(0.9, 0.5), fontsize=8)
    axs[0, 0].axis("off")

    for i_dof in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        axs[num_line, num_col].step(
            time_control_without,
            taudot_without[i_dof],
            color="tab:blue",
            alpha=0.75,
            linewidth=1,
            label="Kinematic tucking constraints",
        )
        axs[num_line, num_col].step(
            time_control_CL,
            taudot_CL[i_dof],
            color="tab:orange",
            alpha=0.75,
            linewidth=1,
            label="Holonomic tucking constraints",
        )

        plot_vertical_time_lines(
            time_end_phase_CL[0],
            time_end_phase_without[0],
            axs[num_line, num_col],
            color="k",
            linestyle="-",
            linewidth=0.5,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[1],
            time_end_phase_without[1],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[1],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[2],
            time_end_phase_without[2],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[2],
            linewidth=None,
        )
        plot_vertical_time_lines(
            time_end_phase_CL[3],
            time_end_phase_without[3],
            axs[num_line, num_col],
            color=None,
            linestyle=phase_delimiter[3],
            linewidth=None,
        )

        axs[num_line, num_col].set_title(dof_names_tau[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque derivative [Nm/s]", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Joint torque derivative [Nm/s]", fontsize=7)  # Leg Rotation
    # axs[0, 2].set_yticklabels([])
    # axs[1, 1].set_yticklabels([])
    # axs[1, 2].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("taudot" + "." + format_graph, format=format_graph)


if PLOT_INERTIA_FLAG:
    # Inertia
    inertia_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    inertia_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        inertia_CL[i, :] = np.diagonal(model.bodyInertia(data_CL["q_all"][:, i]).to_array()).T
        inertia_without[i, :] = np.diagonal(model.bodyInertia(data_without["q_all"][:, i]).to_array()).T

    fig, ax = plt.subplots(4, 1, figsize=(8, 9))
    ax[0].plot(
        time_vector_without,
        inertia_without[:, 0],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[0].plot(
        time_vector_CL,
        inertia_CL[:, 0],
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], ax[0], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1], time_end_phase_without[1], ax[0], color=None, linestyle=phase_delimiter[1], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2], time_end_phase_without[2], ax[0], color=None, linestyle=phase_delimiter[2], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3], time_end_phase_without[3], ax[0], color=None, linestyle=phase_delimiter[3], linewidth=None
    )

    ax[0].set_ylabel("Moment of inertia\n" + r"[$kg.m^2$]")
    ax[0].set_xlim(time_min_graph, time_max_graph)
    ax[0].grid(True, linewidth=0.4)
    ax[0].legend(bbox_to_anchor=(0.95, 1.45), ncol=2)

    # Angular momentum
    ang_mom_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    ang_mom_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        ang_mom_CL[i, :] = model.angularMomentum(data_CL["q_all"][:, i], data_CL["qdot_all"][:, i]).to_array()
        ang_mom_without[i, :] = model.angularMomentum(
            data_without["q_all"][:, i], data_without["qdot_all"][:, i]
        ).to_array()

    ax[1].plot(
        time_vector_without,
        ang_mom_without[:, 0],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[1].plot(
        time_vector_CL,
        ang_mom_CL[:, 0],
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], ax[1], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1], time_end_phase_without[1], ax[1], color=None, linestyle=phase_delimiter[1], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2], time_end_phase_without[2], ax[1], color=None, linestyle=phase_delimiter[2], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3], time_end_phase_without[3], ax[1], color=None, linestyle=phase_delimiter[3], linewidth=None
    )

    ax[1].set_ylabel("Angular momentum\n" + r"[$kg.m^2/s$]")
    ax[1].set_xlim(time_min_graph, time_max_graph)
    ax[1].grid(True, linewidth=0.4)

    # Body velocity
    body_velo_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    body_velo_without = np.zeros((data_without["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        body_velo_CL[i, :] = model.bodyAngularVelocity(data_CL["q_all"][:, i], data_CL["qdot_all"][:, i]).to_array()
        body_velo_without[i, :] = model.bodyAngularVelocity(
            data_without["q_all"][:, i], data_without["qdot_all"][:, i]
        ).to_array()

    ax[2].plot(
        time_vector_without,
        body_velo_without[:, 0],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[2].plot(
        time_vector_CL,
        body_velo_CL[:, 0],
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], ax[2], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1], time_end_phase_without[1], ax[2], color=None, linestyle=phase_delimiter[1], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2], time_end_phase_without[2], ax[2], color=None, linestyle=phase_delimiter[2], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3], time_end_phase_without[3], ax[2], color=None, linestyle=phase_delimiter[3], linewidth=None
    )

    ax[2].set_ylabel("Somersault velocity\n" + r"[$rad/s$]")
    ax[2].set_xlim(time_min_graph, time_max_graph)
    ax[2].grid(True, linewidth=0.4)

    # Centrifugal effect
    centricugal_CL = model.mass() * body_velo_CL[:, 0] ** 2 * np.sqrt(inertia_CL[:, 0] / model.mass())
    centricugal_without = model.mass() * body_velo_without[:, 0] ** 2 * np.sqrt(inertia_without[:, 0] / model.mass())

    ax[3].plot(
        time_vector_without,
        centricugal_without,
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[3].plot(
        time_vector_CL,
        centricugal_CL,
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], ax[3], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1], time_end_phase_without[1], ax[3], color=None, linestyle=phase_delimiter[1], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2], time_end_phase_without[2], ax[3], color=None, linestyle=phase_delimiter[2], linewidth=None
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3], time_end_phase_without[3], ax[3], color=None, linestyle=phase_delimiter[3], linewidth=None
    )

    ax[3].set_ylabel("Centrifugal pseudo-force\n" + r"[$N$]")
    ax[3].set_xlim(time_min_graph, time_max_graph)
    ax[3].grid(True, linewidth=0.4)
    ax[3].set_xlabel("Time [s]")

    fig.subplots_adjust()
    plt.savefig("Inertia" + "." + format_graph, format = format_graph)
    # plt.show()


if PLOT_ENERY_FLAG:
    power_without = np.abs(tau_without * qdot_without_rad[3:, :])
    power_CL = np.abs(tau_CL * qdot_CL_rad[3:, :])
    power_total_without = np.sum(power_without, axis=0)
    power_total_CL = np.sum(power_CL, axis=0)
    energy_without = np.trapz(power_total_without, time_vector_without)
    energy_CL = np.trapz(power_total_CL, time_vector_CL)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(time_vector_CL, power_total_CL, color="tab:orange", label="Holonomic tucking constraints")
    axs[0].plot(time_vector_without, power_total_without, color="tab:blue", label="Kinematic tucking constraints")

    plot_vertical_time_lines(
        time_end_phase_CL[0], time_end_phase_without[0], axs[0], color="k", linestyle="-", linewidth=0.5
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1],
        time_end_phase_without[1],
        axs[0],
        color=None,
        linestyle=phase_delimiter[1],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2],
        time_end_phase_without[2],
        axs[0],
        color=None,
        linestyle=phase_delimiter[2],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3],
        time_end_phase_without[3],
        axs[0],
        color=None,
        linestyle=phase_delimiter[3],
        linewidth=None,
    )

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
    plt.show()
