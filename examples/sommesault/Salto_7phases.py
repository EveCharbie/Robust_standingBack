"""
The aim of this code is to create a movement a salto in 7 phases with a 2D model.

Phase 0: Preparation propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objective(s) function(s): minimize time, tau and qdot derivative

Phase 1: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contacts (TOE_Y, TOE_Z)
- Objective(s) function(s): minimize time, velocity of CoM at the end of the phase, tau and qdot derivative

Phase 2: Wait phase (Flight phase)
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 3: Flight phase
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s):0 contact
- Objective(s) function(s): maximize heigh CoM, time, and minimize tau and qdot derivative

Phase 4: Tucked phase
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): 0 contact
- Objective(s) function(s): minimize tau and qdot derivative

Phase 5: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): 0 contact
- Objective(s) function(s): minimize tau and qdot derivative

Phase 6: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objective(s) function(s): minimize velocity CoM at the end, minimize tau and qdot derivative

"""

# --- Import package --- #
import numpy as np
import pickle
from Save import save_results_with_pickle
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    Axis,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)


# --- Parameters --- #
movement = "Salto_Test_with_elbow"
version = 8
nb_phase = 7
Interpolation_lineaire = True
name_folder_model = "../models"


# --- Save results --- #
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
    states_all = []
    tau = []

    for i in range(len(sol.states)):
        q.append(sol.states[i]["q"])
        qdot.append(sol.states[i]["qdot"])
        states_all.append(sol.states[i]["all"])
        tau.append(sol.controls[i]["tau"])

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    data["detailed_cost"] = sol.detailed_cost
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
        BiorbdModel(biorbd_model_path[6]),
    )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.8 for i in tau_min_total]
    tau_max = [i * 0.8 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Preparation propulsion):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.01, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=0, derivative=True)

    # Phase 1 (Propulsion): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=1, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=1, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=1, derivative=True)

    # Phase 2 (Wait phase, flight phase): Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, phase=2, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, derivative=True, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=2, derivative=True)

    # Phase 3 (Flight phase): Max time and height CoM
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=3, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 4 (Tucked phase):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=4, min_bound=0.2, max_bound=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.ALL_SHOOTING, weight=-10000, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=4, derivative=True)

    # Phase 5 (Preparation landing): Minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=5, min_bound=0.001, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=5, derivative=True)

    # Phase 6 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=6, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=6, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=6, derivative=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=10000000, phase=6, axes=Axis.Y
    )

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Phase 0: Preparation propulsion (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
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
        node=Node.START,
        contact_index=2,
        phase=0,
    )

    # Phase 1: Propulsion (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
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

    # Phase 4: Tucked phase (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.MID,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=4,
    )

    # Phase 6: Landing (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=6,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=6,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=6,
    )

    # Transition phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=5)

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [0.0188, 0.1368, -0.1091, 1.78, 0.5437, 0.191, -0.1452, 0.1821]
    pose_propulsion_start = [-0.2347, -0.4555, -0.8645, 0.4821, 0.03, 2.5905, -2.2897, 0.5538]
    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.03, 0.5353, -0.8367, 0.1119]
    pose_takeout_end = [-0.2803, 0.4015, 0.5049, 3.0558, 1.7953, 0.2255, -0.3913, -0.575]
    pose_salto_start = [-0.3269, 0.6814, 0.9003, 0.35, 1.43, 2.3561, -2.3000, 0.6999]
    pose_salto_end = [-0.8648, 1.3925, 3.7855, 0.35, 1.14, 2.3561, -2.3000, 0.6999]
    pose_landing_start = [-0.9554, 0.1588, 5.8322, -0.4561, 0.03, 0.6704, -0.5304, 0.6546]
    pose_landing_end = [-0.9461, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]  # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Preparation propulsion
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_at_first_node  # impose the first position
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot
    x_bounds[0]["q"].min[0, 2] = -1
    x_bounds[0]["q"].max[0, 2] = 1

    # Phase 1: Propulsion
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, 1] = -np.pi / 2
    x_bounds[1]["q"].max[2, 1] = np.pi / 2
    x_bounds[1]["q"].min[0, 2] = -1
    x_bounds[1]["q"].max[0, 2] = 1
    x_bounds[1]["q"].min[6, 2] = -np.pi / 8
    x_bounds[1]["q"].max[6, 2] = 0
    x_bounds[1]["q"].min[5, 2] = -np.pi / 8
    x_bounds[1]["q"].max[5, 2] = 0

    # Phase 2: Flight phase (Waiting phase)
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[2, 1] = -np.pi / 2
    x_bounds[2]["q"].max[2, 1] = 2 * np.pi
    x_bounds[2]["q"].min[0, 2] = -1
    x_bounds[2]["q"].max[0, 2] = 1
    x_bounds[2]["q"].min[6, :] = -np.pi / 8
    x_bounds[2]["q"].max[6, :] = 0
    x_bounds[2]["q"].min[5, :] = -np.pi / 8
    x_bounds[2]["q"].max[5, :] = 0

    # Phase 3: Flight phase
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].min[2, 1] = -np.pi / 2
    x_bounds[3]["q"].max[2, 1] = 2 * np.pi
    x_bounds[3]["q"].min[0, 2] = -1
    x_bounds[3]["q"].max[0, 2] = 1

    # Phase 4: Tucked phase
    x_bounds.add("q", bounds=bio_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=bio_model[4].bounds_from_ranges("qdot"), phase=4)
    x_bounds[4]["q"].min[2, 1] = -np.pi / 2
    x_bounds[4]["q"].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[4]["q"].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[4]["q"].max[2, 2] = 2 * np.pi + 0.5
    x_bounds[4]["q"].min[6, :] = -2.3
    x_bounds[4]["q"].max[6, :] = -np.pi / 4
    x_bounds[4]["q"].min[5, :] = 0
    x_bounds[4]["q"].max[5, :] = 3 * np.pi / 4
    x_bounds[4]["q"].min[0, 2] = -1
    x_bounds[4]["q"].max[0, 2] = 1

    # Phase 5: Preparation landing
    x_bounds.add("q", bounds=bio_model[5].bounds_from_ranges("q"), phase=5)
    x_bounds.add("qdot", bounds=bio_model[5].bounds_from_ranges("qdot"), phase=5)
    x_bounds[5]["q"].min[2, :] = -np.pi / 2
    x_bounds[5]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[5]["q"].min[0, 2] = -1
    x_bounds[5]["q"].max[0, 2] = 1

    # Phase 6: Landing
    x_bounds.add("q", bounds=bio_model[6].bounds_from_ranges("q"), phase=6)
    x_bounds.add("qdot", bounds=bio_model[6].bounds_from_ranges("qdot"), phase=6)
    x_bounds[6]["q"].min[2, :] = 2 * np.pi - 1.5
    x_bounds[6]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[6]["q"].min[0, 2] = -1
    x_bounds[6]["q"].max[0, 2] = 1
    x_bounds[6]["q"][:, 2] = pose_landing_end
    x_bounds[6]["qdot"][:, 2] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(
        "q", np.array([pose_at_first_node, pose_propulsion_start]).T, interpolation=InterpolationType.LINEAR, phase=0
    )
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add(
        "q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR, phase=1
    )
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", np.array(pose_at_first_node), phase=2)
    x_init.add("qdot", np.array([0] * n_qdot), phase=2)
    x_init.add("q", np.array([pose_takeout_end, pose_salto_start]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q", np.array([pose_salto_start, pose_salto_end]).T, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=5)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=5)
    x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR, phase=6)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=6)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=0,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=1,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=2,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=3,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=4,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=5,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=6,
    )

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[1].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[2].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[3].nb_tau - 3), phase=3)
    u_init.add("tau", [tau_init] * (bio_model[4].nb_tau - 3), phase=4)
    u_init.add("tau", [tau_init] * (bio_model[5].nb_tau - 3), phase=5)
    u_init.add("tau", [tau_init] * (bio_model[6].nb_tau - 3), phase=6)

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        n_threads=32,
    )


# --- Load model --- #
def main():
    ocp = prepare_ocp(
        biorbd_model_path=(
            str(name_folder_model) + "/" + "Model2D_8Dof_3C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_2C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_3C_5M.bioMod",
        ),
        phase_time=(0.2, 0.05, 0.05, 0.05, 0.4, 0.05, 0.2),
        n_shooting=(20, 5, 5, 5, 40, 5, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)

    # --- Show results --- #
    save_results_with_pickle(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
