import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("../holonomic_research/")
from plot_actuators import actuator_function, Joint


# --- Save --- #
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


# --- Graph --- #
def graph_all_comparaison(data_CL, data_without, format_graph="svg"):

    n_shooting_plus_one = (21, 21, 31, 31, 31)
    phase_delimiter = ["-", "--", ":", "-.", "-"]

    # Solution with closed-loop constraints
    # lambdas = data_CL["lambda"]
    q_CL_rad = data_CL["q_all"][:, :]
    q_CL_rad_without_last_node = np.hstack(
        (
            data_CL["q_all"][:, :20],
            data_CL["q_all"][:, 21 : 21 + 20],
            data_CL["q_all"][:, 21 + 21 : 21 + 21 + 30],
            data_CL["q_all"][:, 21 + 21 + 31 : 21 + 21 + 31 + 30],
            data_CL["q_all"][:, 21 + 21 + 31 + 31 : 21 + 21 + 31 + 31 + 30],
        )
    )
    q_CL_rad_with_last_node = data_CL["q_all"]
    q_CL_rad[6, :] = q_CL_rad[6, :] * -1
    q_CL_deg = np.vstack([q_CL_rad[0:2, :], q_CL_rad[2:, :] * 180 / np.pi])
    qdot_CL_rad = data_CL["qdot_all"]
    qdot_CL_rad[6, :] = qdot_CL_rad[6, :] * -1
    qdot_CL_deg = np.vstack([qdot_CL_rad[0:2, :], qdot_CL_rad[2:, :] * 180 / np.pi])
    tau_CL = data_CL["tau_all"]
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
    time_end_phase_tau_CL = np.array(time_end_phase_tau_CL) - time_CL[0][-1]
    # time_end_phase_pourcentage_CL = time_to_percentage(np.vstack(time_end_phase_CL))
    # time_end_phase_tau_pourcentage_CL = time_to_percentage(np.vstack(time_end_phase_tau_CL))

    # time_pourcentage_CL = time_to_percentage(time_CL)
    time_tau_CL = (
        np.vstack([arr[:-1, :] for arr in time_CL]).reshape(
            -1,
        )
        - time_CL[0][-1]
    )
    # time_tau_pourcentage_CL = time_to_percentage(time_tau_CL)

    # Solution without closed-loop constraints
    q_without_rad = data_without["q_all"][:, :]
    q_without_rad_without_last_node = np.hstack(
        (
            data_without["q_all"][:, :20],
            data_without["q_all"][:, 21 : 21 + 20],
            data_without["q_all"][:, 21 + 21 : 21 + 21 + 30],
            data_without["q_all"][:, 21 + 21 + 31 : 21 + 21 + 31 + 30],
            data_without["q_all"][:, 21 + 21 + 31 + 31 : 21 + 21 + 31 + 31 + 30],
        )
    )
    q_without_rad[6, :] = q_without_rad[6, :] * -1
    q_without_deg = np.vstack([q_without_rad[0:2, :], q_without_rad[2:, :] * 180 / np.pi])
    qdot_without_rad = data_without["qdot_all"]
    qdot_without_rad[6, :] = qdot_without_rad[6, :] * -1
    qdot_without_deg = np.vstack([qdot_without_rad[0:2, :], qdot_without_rad[2:, :] * 180 / np.pi])
    tau_without = data_without["tau_all"]

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
    time_end_phase_tau_without = np.array(time_end_phase_tau_without) - time_without[0][-1]
    # time_end_phase_pourcentage_without = time_to_percentage(np.vstack(time_end_phase_without))

    # time_pourcentage_without = time_to_percentage(time_without)
    time_tau_without = (
        np.vstack([arr[:-1, :] for arr in time_without]).reshape(
            -1,
        )
        - time_without[0][-1]
    )
    # time_tau_pourcentage_without = time_to_percentage(time_tau_without)

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
    time_max_graph = max(time_vector_without[-1], time_vector_CL[-1])
    time_min_graph = min(time_vector_without[0], time_vector_CL[0])
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

    # Theoretical min and max bound on tau based on actuators

    actuators = {
        "Shoulders": Joint(
            tau_max_plus=112.8107 * 2,
            theta_opt_plus=-41.0307 * np.pi / 180,
            r_plus=109.6679 * np.pi / 180,
            tau_max_minus=162.7655 * 2,
            theta_opt_minus=-101.6627 * np.pi / 180,
            r_minus=103.9095 * np.pi / 180,
            min_q=-0.7,
            max_q=3.1,
        ),
        "Elbows": Joint(
            tau_max_plus=80 * 2,
            theta_opt_plus=np.pi / 2 - 0.1,
            r_plus=40 * np.pi / 180,
            tau_max_minus=50 * 2,
            theta_opt_minus=np.pi / 2 - 0.1,
            r_minus=70 * np.pi / 180,
            min_q=0,
            max_q=2.09,
        ),  # this one was not measured, I just tried to fit https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
        "Hips": Joint(
            tau_max_plus=220.3831 * 2,
            theta_opt_plus=25.6939 * np.pi / 180,
            r_plus=56.4021 * np.pi / 180,
            tau_max_minus=490.5938 * 2,
            theta_opt_minus=72.5836 * np.pi / 180,
            r_minus=48.6999 * np.pi / 180,
            min_q=-0.4,
            max_q=2.6,
        ),
        "Knees": Joint(
            tau_max_plus=367.6643 * 2,
            theta_opt_plus=-61.7303 * np.pi / 180,
            r_plus=31.7218 * np.pi / 180,
            tau_max_minus=177.9694 * 2,
            theta_opt_minus=-33.2908 * np.pi / 180,
            r_minus=57.0370 * np.pi / 180,
            min_q=-2.3,
            max_q=0.02,
        ),
        "Ankles": Joint(
            tau_max_plus=153.8230 * 2,
            theta_opt_plus=0.7442 * np.pi / 180,
            r_plus=58.9832 * np.pi / 180,
            tau_max_minus=171.9903 * 2,
            theta_opt_minus=12.6824 * np.pi / 180,
            r_minus=21.8717 * np.pi / 180,
            min_q=-0.7,
            max_q=0.7,
        ),
    }

    tau_CL_min_bound = np.zeros((5, tau_CL.shape[1]))
    tau_CL_max_bound = np.zeros((5, tau_CL.shape[1]))
    tau_without_min_bound = np.zeros((5, tau_without.shape[1]))
    tau_without_max_bound = np.zeros((5, tau_without.shape[1]))
    for i_dof, key in enumerate(actuators.keys()):
        tau_CL_min_bound[i_dof, :] = -actuator_function(
            actuators[key].tau_max_minus,
            actuators[key].theta_opt_minus,
            actuators[key].r_minus,
            # q_CL_rad_without_last_node[i_dof + 3],
            q_CL_rad_with_last_node[i_dof + 3],
        )
        tau_CL_max_bound[i_dof, :] = actuator_function(
            actuators[key].tau_max_plus,
            actuators[key].theta_opt_plus,
            actuators[key].r_plus,
            # q_CL_rad_without_last_node[i_dof + 3],
            q_CL_rad_with_last_node[i_dof + 3],
        )
        tau_without_min_bound[i_dof, :] = -actuator_function(
            actuators[key].tau_max_minus,
            actuators[key].theta_opt_minus,
            actuators[key].r_minus,
            # q_without_rad_without_last_node[i_dof + 3],
            q_CL_rad_with_last_node[i_dof + 3],
        )
        tau_without_max_bound[i_dof, :] = actuator_function(
            actuators[key].tau_max_plus,
            actuators[key].theta_opt_plus,
            actuators[key].r_plus,
            # q_without_rad_without_last_node[i_dof + 3],
            q_CL_rad_with_last_node[i_dof + 3],
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
        [], [], [], color="tab:orange", alpha=0.1, label=r"$\tilde{\tau}_J$ Holonomic tucking contraints", linewidth=0.5
    )
    axs[0, 0].fill_between(
        [], [], [], color="tab:blue", alpha=0.1, label=r"$\tilde{\tau}_J$ Kinematic tucking constraints", linewidth=0.5
    )
    axs[0, 0].legend(loc="center right", bbox_to_anchor=(0.9, 0.5), fontsize=8)
    axs[0, 0].axis("off")

    for i_dof in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        # axs[num_line, num_col].step(range(len(tau_without[i_dof])), tau_without_max_bound[i_dof], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(
            # time_tau_without,
            time_vector_without,
            tau_without_max_bound[i_dof],
            np.ones(tau_without_max_bound[i_dof].shape) * 1000,
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        # axs[num_line, num_col].step(range(len(tau_without[i_dof])), tau_without_min_bound[i_dof], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(
            time_vector_without,
            # time_tau_CL,
            np.ones(tau_without_max_bound[i_dof].shape) * -1000,
            tau_without_min_bound[i_dof],
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        # axs[num_line, num_col].step(range(len(tau_CL[i_dof])), tau_CL_max_bound[i_dof], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(
            # time_tau_without,
            time_vector_without,
            tau_CL_max_bound[i_dof],
            np.ones(tau_without_max_bound[i_dof].shape) * 1000,
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )
        # axs[num_line, num_col].step(range(len(tau_CL[i_dof])), tau_CL_min_bound[i_dof], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(
            time_vector_without,
            np.ones(tau_without_max_bound[i_dof].shape) * -1000,
            tau_CL_min_bound[i_dof],
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=0.5,
        )

        axs[num_line, num_col].step(
            time_vector_without,
            tau_without[i_dof],
            color="tab:blue",
            alpha=0.75,
            linewidth=1,
            label="Kinematic tucking constraints",
            where="mid",
        )
        axs[num_line, num_col].step(
            time_vector_without,
            tau_CL[i_dof],
            color="tab:orange",
            alpha=0.75,
            linewidth=1,
            label="Holonomic tucking constraints",
            where="mid",
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
    dt_CL = np.vstack(
        (
            time_CL[0][1:] - time_CL[0][:-1],
            time_CL[1][1:] - time_CL[1][:-1],
            time_CL[2][1:] - time_CL[2][:-1],
            time_CL[3][1:] - time_CL[3][:-1],
            time_CL[4][1:] - time_CL[4][:-1],
        )
    )
    dt_without = np.vstack(
        (
            time_without[0][1:] - time_without[0][:-1],
            time_without[1][1:] - time_without[1][:-1],
            time_without[2][1:] - time_without[2][:-1],
            time_without[3][1:] - time_without[3][:-1],
            time_without[4][1:] - time_without[4][:-1],
        )
    )
    tau_CL_ratio_all = np.zeros(tau_CL.shape)
    tau_without_ratio_all = np.zeros(tau_without.shape)

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    axs[0, 0].step(
        # time_tau_CL,
        time_vector_CL,
        np.sum(np.abs(tau_CL), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[0, 0].step(
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
    axs[1, 0].step(
        time_vector_CL,
        np.sum(np.abs(tau_CL_ratio_all), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[1, 0].step(
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
    HARD_CODED_OFFSET = -5
    sum_tau_all_CL = np.sum(np.abs(tau_CL[:, :HARD_CODED_OFFSET] * dt_CL.T))
    sum_tau_all_without = np.sum(np.abs(tau_without[:, :HARD_CODED_OFFSET] * dt_without.T))
    axs[0, 1].bar([0, 1], [sum_tau_all_CL, sum_tau_all_without], color=["tab:orange", "tab:blue"])
    sum_tau_all_ratio_CL = np.sum(np.abs(tau_CL_ratio_all[:, :HARD_CODED_OFFSET] * dt_CL.T))
    sum_tau_all_ratio_without = np.sum(np.abs(tau_without_ratio_all[:, :HARD_CODED_OFFSET] * dt_without.T))
    axs[1, 1].bar([0, 1], [sum_tau_all_ratio_CL, sum_tau_all_ratio_without], color=["tab:orange", "tab:blue"])

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
    # axs[1, 1].fill_between([], [], [], color="tab:orange", label=r"$\int{| \tau / \tilde{\tau}_J | dt}$ HTC")
    # axs[1, 1].fill_between([], [], [], color="tab:blue", label=r"$\int{| \tau / \tilde{\tau}_J | dt}$ KTC")
    axs[1, 1].legend(loc="center right", bbox_to_anchor=(0.5, 3.1), ncol=2)

    axs[1, 1].set_ylabel(r"$\int{ | \tau/{max_\tau} | dt}$" + "\n[s]")
    plt.subplots_adjust(hspace=0.25, wspace=0.4, top=0.95)
    plt.savefig("tau_ratio_all" + "." + format_graph, format=format_graph)
    # plt.show()
