"""
The aim of this code is to create a backward salto with two different technique:
- the first one with a wait phase to simulate an error timing before the salto
- the second technique with the optimal technique

Phase 0: Preparation propulsion
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize time, tau and qdot derivative
- Dynamics: with_contact

Phase 1: Propulsion
- 2 contacts (TOE_Y, TOE_Z)
- Objectives functions: minimize time, velocity of CoM at the end of the phase, tau and qdot derivative
- Dynamics: with_contact

Phase 2: Wait phase (Take-off phase)
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 3: Take-off phase
- 0 contact
- Objectives functions: maximize heigh CoM, time, and minimize tau and qdot derivative

Phase 4: Salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 5: Take-off after salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 6: Landing
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize velocity CoM at the end, minimize tau and qdot derivative

Phase 7: Take-off phase
- 0 contact
- Objectives functions: maximize heigh CoM, time, and minimize tau and qdot derivative

Phase 8: Salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 9: Take-off after salto
- 0 contact
- Objectives functions: maximize max time, minimize tau and qdot derivative

Phase 10: Landing
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize velocity CoM at the end, minimize tau and qdot derivative

"""

# --- Import package --- #

from visualisation import visualisation_movement
import numpy as np
import pickle
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    Axis,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    PhaseTransitionList,
    PhaseTransitionFcn,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    MultinodeObjectiveList,
    MultinodeConstraintFcn,
    MultinodeObjectiveFcn,
)

def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas_q = data_tmp["q"]
    data_time_node = data_tmp["time"]
    datas_qdot = data_tmp["qdot"]
    data_tau = data_tmp["tau"]

    return datas_q, datas_qdot, data_tau, data_time_node

def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
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

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


# --- Parameters --- #
nb_phase = 6
movement = "Salto"
version = 16
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"
pickle_sol_init = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V10.pkl"
q_init, qdot_init, tau_init, time_init = get_created_data_from_pickle(pickle_sol_init)
phase_time_init = []
for i in range(len(time_init)):
    time_final = time_init[i][-1] - time_init[i][0]
    phase_time_init.append(time_final)

n_shooting_init = []
for i in range(len(q_init)):
    n_shooting_final = q_init[i].shape[1] - 1
    n_shooting_init.append(n_shooting_final)


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):
    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
        BiorbdModel(biorbd_model_path[4]),
        BiorbdModel(biorbd_model_path[5]),
    )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]  # with elbow
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]  # with elbow
    tau_min = [i * 0.8 for i in tau_min_total]
    tau_max = [i * 0.8 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives = MultinodeObjectiveList()


    # Phase 0 (Preparation propulsion): Minimize tau and qdot, minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.01, max_bound=0.5, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.001, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=0)

    # Phase 1 (Propulsion): Maximize velocity CoM + Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.01, max_bound=0.2, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=1)

    # Phase 2 (Take-off): Maximize time and height CoM + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, min_bound=0.1, max_bound=0.3, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=2)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, derivative=True, phase=2)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
    #     node=Node.END,
    #     first_marker="BELOW_KNEE",
    #     second_marker="CENTER_HAND",
    #     axis=[Axis.Z, Axis.Y],
    #     phase=2,
    # )

    # Phase 3 (Salto):  Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, derivative=True, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.2, max_bound=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=3)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.START, weight=-10000, phase=3)
    # TODO: Essayer avec et sans
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-10000, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1, derivative=True, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=2, weight=-1, derivative=False,
    #                         quadratic=False, phase=3)

    # Phase 4 (Take-off after salto): Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, min_bound=0.1, max_bound=0.3, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=4)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=4)

    # Phase 5 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=1000, phase=5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.3, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=5)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=5)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=1000, axes=Axis.Y, phase=5
    )
    multinode_objectives.add(
        MultinodeConstraintFcn.STATES_EQUALITY,
        nodes_phase=(4, 5),
        nodes=(Node.END, Node.START),
        key="all",
    )
    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=5)

    # --- Constraints ---#
    # Constraints
    #     - contact[0]: Toe_Y
    #     - contact[1]: Toe_Z
    #     - contact[2]: Heel_Z
    constraints = ConstraintList()

    # Phase 0 (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=0,
    )

    # Phase 1
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=1,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=1,
    )

    # Phase 3
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=3)


    # Phase 5 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    # NON_SLIPPING
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=5,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=5,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=5,
    )

    # Transition phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=4)

     # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Position solution
    pose_at_first_node = [0.0188, 0.1368, -0.1091, 1.78, 0.5437, 0.191, -0.1452,
                          0.25]  # Position of segment during first position
    pose_propulsion_start = [0.0195, -0.1714, -0.8568, -0.0782, 0.5437, 2.0522, -1.6462, 0.5296]
    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [0.1987, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]
    pose_landing_end = [0.1987, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]


    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Preparation propulsion
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_at_first_node
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot  # impose the first position

    # Phase 1: Propulsion
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, :] = -np.pi / 2
    x_bounds[1]["q"].max[2, :] = np.pi / 2

    # Phase 2: Take-off phase
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[2, 0] = -np.pi / 2
    x_bounds[2]["q"].max[2, 0] = np.pi / 2
    x_bounds[2]["q"].min[2, 1] = -np.pi / 4
    x_bounds[2]["q"].max[2, 1] = np.pi / 2
    x_bounds[2]["q"].min[2, -1] = np.pi / 2
    x_bounds[2]["q"].max[2, -1] = np.pi

    # Phase 3: salto
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].min[2, 0] = 0
    x_bounds[3]["q"].max[2, 0] = np.pi / 2
    x_bounds[3]["q"].min[2, 1] = np.pi / 8
    x_bounds[3]["q"].max[2, 1] = 2 * np.pi
    x_bounds[3]["q"].min[2, 2] = 3/4 * np.pi
    x_bounds[3]["q"].max[2, 2] = 3/2 * np.pi

    # x_bounds[3]["q"].max[5, :-1] = 2.6  # TODO: Ajouter
    # x_bounds[3]["q"].min[5, :-1] = 1.96
    # x_bounds[3]["q"].max[6, :-1] = -1.72
    # x_bounds[3]["q"].min[6, :-1] = -2.3

    # Phase 4: Take-off after salto
    x_bounds.add("q", bounds=bio_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=bio_model[4].bounds_from_ranges("qdot"), phase=4)
    x_bounds[4]["q"].min[2, :] = 3/4 * np.pi
    x_bounds[4]["q"].max[2, :] = 2 * np.pi + 0.5

    # Phase 5: landing
    x_bounds.add("q", bounds=bio_model[5].bounds_from_ranges("q"), phase=5)
    x_bounds.add("qdot", bounds=bio_model[5].bounds_from_ranges("qdot"), phase=5)
    x_bounds[5]["q"].min[2, 0] = 3/4 * np.pi
    x_bounds[5]["q"].max[2, 0] = 2 * np.pi + 0.5
    x_bounds[5]["q"].min[2, 1:] = 2 * np.pi - 0.5
    x_bounds[5]["q"].max[2, 1:] = 2 * np.pi + 0.5
    x_bounds[5]["q"].min[0, 2] = -2
    x_bounds[5]["q"].max[0, 2] = 0
    x_bounds[5]["q"][:, 2] = pose_landing_end
    x_bounds[5]["qdot"][:, 2] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    # Phase 0 (prepa propulsion)
    x_init.add("q", np.array([pose_at_first_node, pose_propulsion_start]).T, interpolation=InterpolationType.LINEAR,
               phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    # x_init.add("q", q_init[0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("qdot", qdot_init[0], interpolation=InterpolationType.EACH_FRAME, phase=0)

    # Phase 1 (Propulsion)
    x_init.add("q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR,
               phase=1)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    # x_init.add("q", q_init[1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("qdot", qdot_init[1], interpolation=InterpolationType.EACH_FRAME, phase=1)

    # Phase 2 (take-off)
    x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR,
               phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    # x_init.add("q", q_init[2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("qdot", qdot_init[2], interpolation=InterpolationType.EACH_FRAME, phase=2)


    # Phase 3 (salto)
    x_init.add("q", np.array([pose_salto_start, pose_salto_end]).T, interpolation=InterpolationType.LINEAR,
               phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)

    # x_init.add("q", q_init[3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("qdot", qdot_init[3], interpolation=InterpolationType.EACH_FRAME, phase=3)

    # Phase 4 (flight)
    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR,
               phase=4)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=4)
    # x_init.add("q", q_init[4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("qdot", qdot_init[4], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Phase 5 (landing)
    x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR,
               phase=5)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=5)
    # x_init.add("q", q_init[5], interpolation=InterpolationType.EACH_FRAME, phase=5)
    # x_init.add("qdot", qdot_init[5], interpolation=InterpolationType.EACH_FRAME, phase=5)

    # Define control path constraint
    u_bounds = BoundsList()
    for j in range(0, nb_phase):
        u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                     max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=j)

    u_init = InitialGuessList()
    for j in range(0, nb_phase):
        u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=j)
        # u_init.add("tau", tau_init[j][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=j)

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        phase_transitions=phase_transitions,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        assume_phase_dynamics=True,
        n_threads=32,
    )

    # --- Load model --- #


def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V2.bioMod"
    model_path_2contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V2.bioMod"

    ocp = prepare_ocp(
        biorbd_model_path=(
            model_path_2contact,
            model_path_1contact,
            model_path,
            model_path,
            model_path,
            model_path_2contact,
        ),
        phase_time=(0.2, 0.1, 0.1, 0.4, 0.1, 0.2),
        n_shooting=(20, 10, 10, 40, 10, 20),
        min_bound=0.01,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)
    sol.graphs(show_bounds=True)
    sol.print_cost()

    # --- Show/Save results --- #
    name_movement = str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl"
    save_results(sol, name_movement)
    visualisation_movement(name_movement, model_path_2contact)


if __name__ == "__main__":
    main()
