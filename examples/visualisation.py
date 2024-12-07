import bioviz
from scipy.interpolate import interp1d
import numpy as np
from save import get_created_data_from_pickle


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
    Q = np.concatenate(data["q"], axis=1)
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


def visualisation_closed_loop_5phases(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 5 phases
    (preparation propulsion, propulsion, flight, tucked phase, preparation landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_1 = sol.states[1]["q"]
    q_2 = sol.states[2]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[3]["q_u"].shape[1]))
    q_4 = sol.states[4]["q"]
    for i, ui in enumerate(sol.states[3]["q_u"].T):
        # vi = bio_model.compute_v_from_u_numeric(ui, v_init=DM(np.zeros(2))).toarray()
        vi = bio_model[3].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[3].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_2, q_holo, q_4), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_6phases(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 6 phases
    (preparation propulsion, propulsion, flight, tucked phase, preparation landing, landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_1 = sol.states[1]["q"]
    q_2 = sol.states[2]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[3]["q_u"].shape[1]))
    q_4 = sol.states[4]["q"]
    q_5 = sol.states[5]["q"]
    for i, ui in enumerate(sol.states[3]["q_u"].T):
        # vi = bio_model.compute_v_from_u_numeric(ui, v_init=DM(np.zeros(2))).toarray()
        vi = bio_model[3].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[3].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_2, q_holo, q_4, q_5), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop(name_file_movement: str, name_file_model: str):
    """
    Code to visualize a simulation with a holonomic constraints body-body
    Parameters
    ----------
    name_file_movement:
        Path of the movement simulated
    name_file_model:
        Path of the model used

    Returns
    -------

    """
    bio_model = BiorbdModelCustomHolonomic(name_file_model)

    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model,
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
    )
    bio_model.set_holonomic_configuration(
        constraints_list=holonomic_constraints,
        independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )
    data = get_created_data_from_pickle(name_file_movement)
    nb_q = bio_model.nb_q
    for index, arr in enumerate(data["q"]):
        if arr.shape[0] != 8:
            index_holo = index
    q = data["q"]
    q[index_holo] = np.zeros(shape=(nb_q, q[index_holo].shape[1]))
    for i, ui in enumerate(data["q"][index_holo].T):
        vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model.state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q[index_holo][:, i] = qi
    Q = np.concatenate(q, axis=1)
    visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
    visu.load_movement(Q)
    visu.exec()


def visualisation_closed_loop_4phases_propulsion(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 5 phases
    (preparation propulsion, propulsion, flight, tucked phase)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_1 = sol.states[1]["q"]
    q_2 = sol.states[2]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[3]["q_u"].shape[1]))
    for i, ui in enumerate(sol.states[3]["q_u"].T):
        # vi = bio_model.compute_v_from_u_numeric(ui, v_init=DM(np.zeros(2))).toarray()
        vi = bio_model[3].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[3].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_2, q_holo), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_4phases_reception(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 4 phases
    (flight, tucked phase, preparation landing, landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[1]["q_u"].shape[1]))
    q_2 = sol.states[2]["q"]
    q_3 = sol.states[3]["q"]
    for i, ui in enumerate(sol.states[1]["q_u"].T):

        vi = bio_model[1].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[1].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_holo, q_2, q_3), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_5phases_reception(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 5 phases
    (propulsion, flight, tucked phase, preparation landing, landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_1 = sol.states[1]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[2]["q_u"].shape[1]))
    q_3 = sol.states[3]["q"]
    q_4 = sol.states[4]["q"]
    for i, ui in enumerate(sol.states[2]["q_u"].T):
        vi = bio_model[2].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[2].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_holo, q_3, q_4), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_3phases(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 3 phases (flight, tucked phase, preparation landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[1]["q_u"].shape[1]))
    q_2 = sol.states[2]["q"]
    for i, ui in enumerate(sol.states[1]["q_u"].T):
        vi = bio_model[1].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[1].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_holo, q_2), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_1phase(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 1 phase (tucked phase)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q = np.zeros((bio_model.nb_tau, sol.states["q_u"].shape[0]))
    for i, ui in enumerate(sol.states["q_u"]):
        vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model.state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


def visualisation_closed_loop_2phases(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 2 phases (flight and tucked phase)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[0]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[1]["q_u"].shape[1]))
    for i, ui in enumerate(sol.states[1]["q_u"].T):
        vi = bio_model[1].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[1].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_holo), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


def visualisation_closed_loop_2phases_post(bio_model, sol, model_path):
    """
    Code to visualize a simulation composed of 2 phases (tucked phase and preparation landing)
    and with a holonomic constraints body-body
    Parameters
    ----------
    bio_model:
        models of the simulation
    sol:
        The solution to the ocp at the current pool
    model_path:
        Path of the model used

    Returns
    -------

    """
    q_0 = sol.states[1]["q"]
    q_holo = np.zeros((bio_model[1].nb_q, sol.states[0]["q_u"].shape[1]))
    for i, ui in enumerate(sol.states[0]["q_u"].T):
        vi = bio_model[0].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[0].state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_holo, q_0), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()
