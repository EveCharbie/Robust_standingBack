import pickle


def save_results(sol, name_pickle_file):
    """
    Save all the results of the predictive simulation into a pickle file
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
     name_pickle_file: str
        The desired pickle document path
    """

    data = {}
    q = []
    qdot = []
    tau = []

    for i in range(len(sol.states)):
        q.append(sol.states[i]["q"])
        qdot.append(sol.states[i]["qdot"])
        tau.append(sol.controls[i]["tau"])

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phase_time[1:12]
    data["constraints"] = sol.constraints
    data["controls"] = sol.controls
    data["constraints_scaled"] = sol.controls_scaled
    data["n_shooting"] = sol.ns
    data["time"] = sol.time
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{name_pickle_file}", "wb") as file:
        pickle.dump(data, file)


def save_results_CL(sol, name_pickle_file, index_holonomic_constraints:int):
    """
    Save all the results of the predictive simulation into a pickle file
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
     name_pickle_file: str
        The desired pickle document path
     index_holonomic_constraints:
        Index of the phase who contains a holonomic constraint limb-to-limb
    """

    data = {}
    q = []
    qdot = []
    tau = []

    for i in range(len(sol.states)):
        if i == index_holonomic_constraints:
            q.append(sol.states[i]["q_u"])
            qdot.append(sol.states[i]["qdot_u"])
            tau.append(sol.controls[i]["tau"])
        else:
            q.append(sol.states[i]["q"])
            qdot.append(sol.states[i]["qdot"])
            tau.append(sol.controls[i]["tau"])

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phase_time[1:12]
    data["constraints"] = sol.constraints
    data["controls"] = sol.controls
    data["constraints_scaled"] = sol.controls_scaled
    data["n_shooting"] = sol.ns
    data["time"] = sol.time
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{name_pickle_file}", "wb") as file:
        pickle.dump(data, file)


def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data: All the data of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp