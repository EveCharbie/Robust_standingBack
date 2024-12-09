from bioptim import Solution, SolutionMerge
import casadi as cas
import numpy as np
import pickle
import os


# --- Save results --- #
def save_results_holonomic(
    sol,
    *combinatorial_parameters,
    **extra_parameters,
):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """
    biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed = combinatorial_parameters
    index_holo = 2
    biomedel_holo = sol.ocp.nlp[index_holo].model

    # Save path
    save_folder = extra_parameters["save_folder"]
    folder_path = save_folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = f"{folder_path}/sol_{seed}"
    if sol.status == 0:
        file_path += "_CVG.pkl"
    else:
        file_path += "_DVG.pkl"

    # Symbolic variables
    Q_sym = cas.MX.sym("Q_u", 6)
    Qdot_sym = cas.MX.sym("Qdot_u", 6)
    Tau_sym = cas.MX.sym("Tau", 8)
    lagrangian_multipliers_func = cas.Function(
        "Compute_lagrangian_multipliers",
        [Q_sym, Qdot_sym, Tau_sym],
        [biomedel_holo.compute_the_lagrangian_multipliers(Q_sym, Qdot_sym, Tau_sym)],
    )
    q_holo_func = cas.Function(
        "Compute_q_holo",
        [Q_sym],
        [biomedel_holo.state_from_partition(Q_sym, biomedel_holo.compute_v_from_u_explicit_symbolic(Q_sym))],
    )
    Bvu = biomedel_holo.coupling_matrix(q_holo_func(Q_sym))
    vdot = Bvu @ Qdot_sym
    qdot = biomedel_holo.state_from_partition(Qdot_sym, vdot)
    qdot_holo_func = cas.Function(
        "Compute_qdot_holo",
        [Q_sym, Qdot_sym],
        [qdot],
    )

    data = {}
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    qddot = []
    tau = []
    time = []
    min_bounds_q = []
    max_bounds_q = []
    min_bounds_qdot = []
    max_bounds_qdot = []
    min_bounds_tau = []
    max_bounds_tau = []

    if len(sol.ocp.n_shooting) == 1:
        q = states["q_u"]
        qdot = states["q_udot"]
        tau = controls["tau"]
    else:
        for i in range(len(states)):
            if i == index_holo:
                q_u = states[index_holo]["q_u"]
                qdot_u = states[index_holo]["qdot_u"]
                tau_this_time = controls[index_holo]["tau"]
                tau_this_time = np.vstack((np.zeros((3, tau_this_time.shape[1])), tau_this_time))

                q_holo = np.zeros((8, q_u.shape[1]))
                qdot_holo = np.zeros((8, qdot_u.shape[1]))
                for i_node in range(q_u.shape[1]):
                    q_holo[:, i_node] = np.reshape(q_holo_func(q_u[:, i_node]), (8,))
                    qdot_holo[:, i_node] = np.reshape(qdot_holo_func(q_u[:, i_node], qdot_u[:, i_node]), (8,))

                lambdas = np.zeros((2, tau_this_time.shape[1]))
                for i_node in range(tau_this_time.shape[1]):
                    lambdas[:, i_node] = np.reshape(
                        lagrangian_multipliers_func(q_u[:, i_node], qdot_u[:, i_node], tau_this_time[:, i_node]), (2,)
                    )

                q.append(q_holo)
                qdot.append(qdot_holo)
                tau.append(controls[i]["tau"])
                time.append(list_time[i])
                min_bounds_q.append(sol.ocp.nlp[i].x_bounds["q_u"].min)
                max_bounds_q.append(sol.ocp.nlp[i].x_bounds["q_u"].max)
                min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot_u"].min)
                max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot_u"].max)
                min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
                max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)
            else:
                q.append(states[i]["q"])
                qdot.append(states[i]["qdot"])
                tau.append(controls[i]["tau"])
                time.append(list_time[i])
                min_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].min)
                max_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].max)
                min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].min)
                max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].max)
                min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
                max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)

    data["q"] = q
    data["qdot"] = qdot
    data["qddot"] = qddot
    data["tau"] = tau
    data["time"] = time
    data["min_bounds_q"] = min_bounds_q
    data["max_bounds_q"] = max_bounds_q
    data["min_bounds_qdot"] = min_bounds_qdot
    data["max_bounds_qdot"] = max_bounds_qdot
    data["min_bounds_tau"] = min_bounds_q
    data["max_bounds_tau"] = max_bounds_q
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.add_detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phases_dt
    data["constraints"] = sol.constraints
    data["n_shooting"] = sol.ocp.n_shooting
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x
    data["phase_time"] = sol.ocp.phase_time
    data["dof_names"] = sol.ocp.nlp[0].dof_names
    data["q_all"] = np.hstack(data["q"])
    data["qdot_all"] = np.hstack(data["qdot"])
    data["tau_all"] = np.hstack(data["tau"])
    time_end_phase = []
    time_total = 0
    time_all = []
    for i in range(len(data["time"])):
        time_all.append(data["time"][i] + time_total)
        time_total = time_total + data["time"][i][-1]
        time_end_phase.append(time_total)
    data["time_all"] = np.vstack(time_all)
    data["time_total"] = time_total
    data["time_end_phase"] = time_end_phase

    # Only for the tucked phase
    data["q_u"] = q_u
    data["qdot_u"] = qdot_u
    data["lambda"] = lambdas

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def save_results_taudot(
    sol,
    *combinatorial_parameters,
    **extra_parameters,
):

    biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]
    folder_path = save_folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = f"{folder_path}/sol_{seed}"
    if sol.status == 0:
        file_path += "_CVG.pkl"
    else:
        file_path += "_DVG.pkl"

    data = {}
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    tau = []
    taudot = []
    time = []
    min_bounds_q = []
    max_bounds_q = []
    min_bounds_qdot = []
    max_bounds_qdot = []
    min_bounds_tau = []
    max_bounds_tau = []
    min_bounds_taudot = []
    max_bounds_taudot = []

    for i in range(len(states)):
        q.append(states[i]["q"])
        qdot.append(states[i]["qdot"])
        tau.append(states[i]["tau"])
        taudot.append(controls[i]["taudot"])
        time.append(list_time[i])
        min_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].min)
        max_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].max)
        min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].min)
        max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].max)
        min_bounds_tau.append(sol.ocp.nlp[i].x_bounds["tau"].min)
        max_bounds_tau.append(sol.ocp.nlp[i].x_bounds["tau"].max)
        min_bounds_taudot.append(sol.ocp.nlp[i].u_bounds["taudot"].min)
        max_bounds_taudot.append(sol.ocp.nlp[i].u_bounds["taudot"].max)

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["taudot"] = taudot
    data["time"] = time
    data["min_bounds_q"] = min_bounds_q
    data["max_bounds_q"] = max_bounds_q
    data["min_bounds_qdot"] = min_bounds_qdot
    data["max_bounds_qdot"] = max_bounds_qdot
    data["min_bounds_tau"] = min_bounds_tau
    data["max_bounds_tau"] = max_bounds_tau
    data["min_bounds_taudot"] = min_bounds_taudot
    data["max_bounds_taudot"] = max_bounds_taudot
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.add_detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phases_dt
    data["constraints"] = sol.constraints
    data["n_shooting"] = sol.ocp.n_shooting
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x
    data["phase_time"] = sol.ocp.phase_time
    data["dof_names"] = sol.ocp.nlp[0].dof_names
    data["q_all"] = np.hstack(data["q"])
    data["qdot_all"] = np.hstack(data["qdot"])
    data["tau_all"] = np.hstack(data["tau"])
    time_end_phase = []
    time_total = 0
    time_all = []
    for i in range(len(data["time"])):
        time_all.append(data["time"][i] + time_total)
        time_total = time_total + data["time"][i][-1]
        time_end_phase.append(time_total)
    data["time_all"] = np.vstack(time_all)
    data["time_total"] = time_total
    data["time_end_phase"] = time_end_phase

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    sol.print_cost()

    return


# tau, no taudot, no close loop
def save_results(
    sol,
    *combinatorial_parameters,
    **extra_parameters,
):
    biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]
    folder_path = save_folder

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = f"{folder_path}/sol_{seed}"
    if sol.status == 0:
        file_path += "_CVG.pkl"
    else:
        file_path += "_DVG.pkl"

    data = {}
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    tau = []
    time = []
    min_bounds_q = []
    max_bounds_q = []
    min_bounds_qdot = []
    max_bounds_qdot = []
    min_bounds_tau = []
    max_bounds_tau = []

    for i in range(len(states)):
        q.append(states[i]["q"])
        qdot.append(states[i]["qdot"])
        tau.append(controls[i]["tau"])
        time.append(list_time[i])
        min_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].min)
        max_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].max)
        min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].min)
        max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].max)
        min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
        max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["time"] = time
    data["min_bounds_q"] = min_bounds_q
    data["max_bounds_q"] = max_bounds_q
    data["min_bounds_qdot"] = min_bounds_qdot
    data["max_bounds_qdot"] = max_bounds_qdot
    data["min_bounds_tau"] = min_bounds_q
    data["max_bounds_tau"] = max_bounds_q
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.add_detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phases_dt
    data["constraints"] = sol.constraints
    data["n_shooting"] = sol.ocp.n_shooting
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x
    data["phase_time"] = sol.ocp.phase_time
    data["dof_names"] = sol.ocp.nlp[0].dof_names
    data["q_all"] = np.hstack(data["q"])
    data["qdot_all"] = np.hstack(data["qdot"])
    data["tau_all"] = np.hstack(data["tau"])
    time_end_phase = []
    time_total = 0
    time_all = []
    for i in range(len(data["time"])):
        time_all.append(data["time"][i] + time_total)
        time_total = time_total + data["time"][i][-1]
        time_end_phase.append(time_total)
    data["time_all"] = np.vstack(time_all)
    data["time_total"] = time_total
    data["time_end_phase"] = time_end_phase

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    return
