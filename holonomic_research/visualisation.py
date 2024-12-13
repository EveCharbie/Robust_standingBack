import bioviz
from scipy.interpolate import interp1d
import numpy as np
from save_load_helpers import get_created_data_from_pickle
import matplotlib.pyplot as plt
import math


# --- Visualisation -- #
def visualisation_model(name_file_model: str):
    """
    Code to visualize the model used
    Parameters
    ----------
    name_file_model: str
        Path of the model

    Returns
    -------

    """
    b = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
    b.exec()


def visualisation_movement(name_file_movement: str, name_file_model: str):
    """
    Code to visualize a movement simulated
    Parameters
    ----------
    name_file_movement: str
        Path of the file who contains the movement simulated
    name_file_model: str
        Path of the model

    Returns
    -------

    """
    data = get_created_data_from_pickle(name_file_movement)
    Q = np.hstack(data["q"])
    visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
    visu.load_movement(Q)
    visu.exec()


def visualisation_dedoublement_phase(name_file_movement: str, name_file_model: str, name_file_model_2: str):
    """
    Code to visualize two simulation in the same windows
    Parameters
    ----------
    name_file_movement: str
        Path of the file who contains the movement simulated
    name_file_model: str
        Path of the model
    name_file_model_2: str
        Path of the second model (for the second simulation)

    Returns
    -------

    """
    q, time_node = get_created_data_from_pickle(name_file_movement)
    if len(q) == 11:
        for i in range(0, len(q)):
            time_node[i] = np.array(time_node[i], dtype=np.float32)
        for i in range(1, len(q)):
            q[i] = q[i][:, 1:]
            time_node[i] = time_node[i][1:]

        # Time simulation salto with errors timing
        Q_1 = np.concatenate((q[0], q[1], q[2], q[3], q[4], q[5], q[6]), axis=1)
        time_Q1 = np.concatenate(
            (time_node[0], time_node[1], time_node[2], time_node[3], time_node[4], time_node[5], time_node[6]),
            axis=0,
        )
        duree_Q1 = [time_Q1[i + 1] - time_Q1[i] for i in range(0, len(time_Q1) - 1)]

        # Change time phase 7 to 11: simulation salto without errors timing
        time_Q7 = time_node[7][0]
        ecart = time_node[1][-1] + time_node[7][1] - time_node[7][0]

        for i in range(7, 11):
            time_node[i] = time_node[i] - (time_Q7 - ecart)
        Q_2 = np.concatenate((q[0], q[1], q[7], q[8], q[9], q[10]), axis=1)
        time_Q2 = np.concatenate((time_node[0], time_node[1], time_node[7], time_node[8], time_node[9], time_node[10]))
        duree_Q2 = [time_Q2[i + 1] - time_Q2[i] for i in range(0, len(time_Q2) - 1)]

        # Interpolation simulation salto with errors timing
        Q1_interpolate = np.zeros(shape=(Q_1.shape[0], int((time_Q1[-1] + 0.01) / 0.01)), dtype=float)
        for nb_Dof in range(Q_1.shape[0]):
            interp_func_Q1 = interp1d(time_Q1, Q_1[nb_Dof, :], kind="linear")
            newy = interp_func_Q1(np.arange(time_Q1[0], time_Q1[-1], 0.01))
            Q1_interpolate[nb_Dof] = newy

        # Interpolation simulation salto without errors timing
        Q2_interpolate = np.zeros(shape=(Q_2.shape[0], int((time_Q2[-1] + 0.01) / 0.01)), dtype=float)
        for nb_Dof in range(Q_2.shape[0]):
            interp_func_Q2 = interp1d(time_Q2, Q_2[nb_Dof, :], kind="linear")
            newy = interp_func_Q2(np.arange(time_Q2[0], time_Q2[-1], 0.01))
            Q2_interpolate[nb_Dof] = newy

        if Q1_interpolate.shape[1] < Q2_interpolate.shape[1]:
            Q_add = np.zeros(shape=(Q_1.shape[0], int(Q2_interpolate.shape[1] - Q1_interpolate.shape[1])), dtype=float)
            for i in range(Q2_interpolate.shape[1] - Q1_interpolate.shape[1]):
                Q_add[:, i] = Q1_interpolate[:, -1]
            Q1_interpolate_new = np.concatenate((Q1_interpolate, Q_add), axis=1)

        if Q2_interpolate.shape[1] < Q1_interpolate.shape[1]:
            Q_add = np.zeros(shape=(Q_1.shape[0], int(Q2_interpolate.shape[1] - Q1_interpolate.shape[1])), dtype=float)
            for i in range(Q2_interpolate.shape[1] - Q1_interpolate.shape[1]):
                Q_add[:, i] = Q1_interpolate[:, -1]
            Q1_interpolate_new = np.concatenate((Q1_interpolate, Q_add), axis=1)

        # Visualisation simulation salto with errors timing
        visu_1 = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
        visu_1.load_movement(Q_1)
        visu_1.exec()

        # Visualisation simulation salto without errors timing
        visu_2 = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
        visu_2.load_movement(Q_2)
        visu_2.exec()

        # Visualisation two simulations
        Q_3 = np.concatenate((Q1_interpolate, Q2_interpolate), axis=0)
        visu_3 = bioviz.Viz(name_file_model_2, show_floor=True, show_meshes=True)
        visu_3.load_movement(Q_3)
        visu_3.exec()


def graph_all(sol):
    data = get_created_data_from_pickle(sol)
    lambdas = data["lambda"]
    q = data["q_all"]
    qdot = data["qdot_all"]
    qddot = data["qddot_all"]
    tau = data["tau_all"]
    dof_names = data["dof_names"]

    # Time
    time_total = data["time_total"]
    time_end_phase = data["time_end_phase"]
    time = data["time_all"]

    # Figure q
    fig, axs = plt.subplots(2, math.ceil(q.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(q)
    y_max = np.max(q)
    for nb_seg in range(q.shape[0]):
        axs[num_line, num_col].plot(time, q[nb_seg])
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase[xline], color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(q.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Positions [m]")
    for ax in axs.flat:
        ax.label_outer()
    # fig.suptitle("Evolution des Q")

    # Figure qdot
    fig, axs = plt.subplots(2, math.ceil(qdot.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(qdot)
    y_max = np.max(qdot)
    for nb_seg in range(qdot.shape[0]):
        axs[num_line, num_col].plot(time, qdot[nb_seg])
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase[xline], color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(qdot.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Velocities [m/s]")
    for ax in axs.flat:
        ax.label_outer()
    # fig.suptitle("Evolution des Qdot")

    # Figure tau
    fig, axs = plt.subplots(2, math.ceil(tau.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(tau)
    y_max = np.max(tau)
    for nb_seg in range(tau.shape[0]):
        axs[num_line, num_col].plot(tau[nb_seg])
        value_xline = 0
        for xline in range(len(time_end_phase)):
            value_xline = value_xline + data["n_shooting"][xline]
            axs[num_line, num_col].axvline(value_xline, color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg + 3])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(tau.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Tau")
    for ax in axs.flat:
        ax.label_outer()
    # fig.suptitle("Evolution des Tau")

    # Figure lambdas
    # name_ylabel=["Effort normal au tibia [N]", "Effort de cisaillement au tibia [N]"]
    # fig, axs = plt.subplots(2, math.ceil(lambdas.shape[0]/2))
    # num_col = 0
    # num_line = 0
    # for nb_seg in range(lambdas.shape[0]):
    #    axs[num_col].plot(lambdas[nb_seg])
    #    axs[num_col].set_ylabel(name_ylabel[num_col])
    #    if num_line == 1:
    #        axs[num_line, num_col].set_xlabel('Time [s]')
    #    num_col = num_col + 1
    # fig.suptitle("Evolution des Lambdas")
    fig = plt.figure()
    plt.plot(lambdas[0], color="r", label=["Normal force"])
    plt.plot(lambdas[1], color="g", label=["Shear force"])
    plt.ylabel("Force on the tibia [N]")
    plt.xlabel("Shooting point")
    plt.legend()
    plt.show()


def graph_all_comparaison(sol, sol2):

    # Sol 1
    data = get_created_data_from_pickle(sol)
    lambdas = data["lambda"]
    q = data["q_all"]
    qdot = data["qdot_all"]
    qddot = data["qddot_all"]
    tau = data["tau_all"]
    dof_names = data["dof_names"]
    time_total = data["time_total"]
    time_end_phase = data["time_end_phase"]
    time = data["time_all"]

    # Sol 2
    data2 = get_created_data_from_pickle(sol2)
    q2 = data2["q_all"]
    qdot2 = data2["qdot_all"]
    tau2 = data2["tau_all"]
    dof_names2 = data2["dof_names"]
    time_total2 = data2["time_total"]
    time_end_phase2 = data2["time_end_phase"]
    time2 = data2["time_all"]

    # Figure q
    fig, axs = plt.subplots(2, math.ceil(q.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(q)
    y_max = np.max(q)
    for nb_seg in range(q.shape[0]):
        axs[num_line, num_col].plot(time, q[nb_seg])
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase[xline], color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(q.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Positions [m]")
    for ax in axs.flat:
        ax.label_outer()

    # Figure qdot
    fig, axs = plt.subplots(2, math.ceil(qdot.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(qdot)
    y_max = np.max(qdot)
    for nb_seg in range(qdot.shape[0]):
        axs[num_line, num_col].plot(time, qdot[nb_seg])
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase[xline], color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(qdot.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Velocities [m/s]")
    for ax in axs.flat:
        ax.label_outer()

    # Figure tau
    fig, axs = plt.subplots(2, math.ceil(tau.shape[0] / 2))
    num_col = 0
    num_line = 0
    y_min = np.min(tau)
    y_max = np.max(tau)
    for nb_seg in range(tau.shape[0]):
        axs[num_line, num_col].plot(tau[nb_seg])
        value_xline = 0
        for xline in range(len(time_end_phase)):
            value_xline = value_xline + data["n_shooting"][xline]
            axs[num_line, num_col].axvline(value_xline, color="k", linestyle="--")
        axs[num_line, num_col].set_title(dof_names[nb_seg + 3])
        axs[num_line, num_col].set_ylim(y_min, y_max)
        num_col = num_col + 1
        if nb_seg == math.ceil(tau.shape[0] / 2) - 1:
            num_col = 0
            num_line = 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Tau")
    for ax in axs.flat:
        ax.label_outer()

    # Figure lambdas
    fig = plt.figure()
    plt.plot(lambdas[0], color="r", label=["Normal force"])
    plt.plot(lambdas[1], color="g", label=["Shear force"])
    plt.ylabel("Force on the tibia [N]")
    plt.xlabel("Shooting point")
    plt.legend()
    plt.show()


def graph_q(bio_model, sol):
    data = get_created_data_from_pickle(sol)
    q_0 = data["q"][0]
    q_1 = data["q"][1]
    q_holo = np.zeros((bio_model[0].nb_q, data["q"][2].shape[1]))
    q_3 = data["q"][3]
    q_4 = data["q"][4]
    for i, ui in enumerate(data["q"][2].T):
        vi = bio_model[2].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[2].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_holo, q_3, q_4), axis=1)
    n_shooting = [q_0.shape[1], q_1.shape[1], q_holo.shape[1], q_3.shape[1], q_4.shape[1]]

    for nb_seg in range(q.shape[0]):
        plt.figure()
        plt.plot(q[nb_seg])
        xline = 0
        for line in range(len(n_shooting)):
            xline = xline + n_shooting[line]
            plt.axvline(xline)
        plt.show()
