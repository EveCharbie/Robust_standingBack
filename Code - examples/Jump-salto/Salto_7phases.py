"""
The aim of this code is to create a movement a salto in 7 phases with a 2D model.


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
- Objectives functions: maximize  time, and minimize tau and qdot derivative




Phase 4: Salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative, maximize heigh CoM




Phase 5: Take-off after salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative




Phase 6: Landing
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize velocity CoM at the end, minimize tau and qdot derivative




"""
# --- Import package --- #
import numpy as np
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
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack-main/Model"


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
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7])

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

    # Phase 2 (Wait phase, take-off): Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, phase=2, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, derivative=True, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=2, derivative=True)

    # Phase 3 (Take-off): Max time and height CoM
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=3, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 4 (Salto):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=4, min_bound=0.2, max_bound=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.ALL_SHOOTING, weight=-10000, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=4, derivative=True)

    # Phase 5 (Take-off after salto): Minimize time
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
        node=Node.START,
        contact_index=2,
        phase=0,
    )

    # Phase 1 (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)

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

    # Phase 4 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.MID,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=4,
    )

    # Phase 6 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
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
    pose_at_first_node = [
        0.0188,
        0.1368,
        -0.1091,
        1.78,
        0.5437,
        0.191,
        -0.1452,
        0.1821,
    ]  # Position of segment during first position
    pose_propulsion_start = [
        -0.2347217373715483,
        -0.45549996131551357,
        -0.8645258574574489,
        0.4820766547674885,
        0.03,
        2.590467089448695,
        -2.289747592408045,
        0.5538056491954265,
    ]
    pose_takeout_start = [
        -0.2777672842694191,
        0.03995514292843797,
        0.1930477703559439,
        2.589642304908377,
        0.03,
        0.5353536016159908,
        -0.8367077461678971,
        0.11196901833050495,
    ]
    pose_takeout_end = [-0.2803, 0.4015, 0.5049, 3.0558, 1.7953, 0.2255, -0.3913, -0.575]
    pose_salto_start = [
        -0.3269534844623969,
        0.681422172573302,
        0.9003344030624946,
        0.35,
        1.43,
        2.3561945135532367,
        -2.300000008273391,
        0.6999999941919349,
    ]
    pose_salto_end = [
        -0.8648803377623905,
        1.3925287774995057,
        3.785530485157555,
        0.35,
        1.14,
        2.3561945105754827,
        -2.300000018314619,
        0.6999999322366998,
    ]
    pose_landing_start = [
        -0.9554004763233065,
        0.15886445602166693,
        5.832254254152056,
        -0.45610833795726297,
        0.03,
        0.6704528346396729,
        -0.5304889643328282,
        0.654641794221728,
    ]
    pose_landing_end = [-0.9461201943294933, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]  # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Preparation propulsion
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot  # impose the first position
    x_bounds[0].min[0, 2] = -1
    x_bounds[0].max[0, 2] = 1

    # Phase 1: Propulsion
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1].min[2, 1] = -np.pi / 2
    x_bounds[1].max[2, 1] = np.pi / 2
    x_bounds[1].min[0, 2] = -1
    x_bounds[1].max[0, 2] = 1
    x_bounds[1].min[6, 2] = -np.pi / 8
    x_bounds[1].max[6, 2] = 0
    x_bounds[1].min[5, 2] = -np.pi / 8
    x_bounds[1].max[5, 2] = 0

    # Phase 2: Take-off phase (Waiting phase)
    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2].min[2, 1] = -np.pi / 2
    x_bounds[2].max[2, 1] = 2 * np.pi
    x_bounds[2].min[0, 2] = -1
    x_bounds[2].max[0, 2] = 1
    x_bounds[2].min[6, :] = -np.pi / 8
    x_bounds[2].max[6, :] = 0
    x_bounds[2].min[5, :] = -np.pi / 8
    x_bounds[2].max[5, :] = 0

    # Phase 3: Take-off phase
    x_bounds.add(bounds=bio_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds[3].min[2, 1] = -np.pi / 2
    x_bounds[3].max[2, 1] = 2 * np.pi
    x_bounds[3].min[0, 2] = -1
    x_bounds[3].max[0, 2] = 1

    # Phase 4: salto
    x_bounds.add(bounds=bio_model[4].bounds_from_ranges(["q", "qdot"]))
    x_bounds[4].min[2, 1] = -np.pi / 2
    x_bounds[4].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[4].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[4].max[2, 2] = 2 * np.pi + 0.5
    x_bounds[4].min[6, :] = -2.3
    x_bounds[4].max[6, :] = -np.pi / 4
    x_bounds[4].min[5, :] = 0
    x_bounds[4].max[5, :] = 3 * np.pi / 4
    x_bounds[4].min[0, 2] = -1
    x_bounds[4].max[0, 2] = 1

    # Phase 5: Take-off after salto
    x_bounds.add(bounds=bio_model[5].bounds_from_ranges(["q", "qdot"]))
    x_bounds[5].min[2, :] = -np.pi / 2
    x_bounds[5].max[2, :] = 2 * np.pi + 0.5
    x_bounds[5].min[0, 2] = -1
    x_bounds[5].max[0, 2] = 1

    # Phase 6: landing
    x_bounds.add(bounds=bio_model[6].bounds_from_ranges(["q", "qdot"]))
    x_bounds[6].min[2, :] = 2 * np.pi - 1.5
    x_bounds[6].max[2, :] = 2 * np.pi + 0.5
    x_bounds[6][:, 2] = pose_landing_end + [0] * n_qdot
    x_bounds[6].min[0, 2] = -1
    x_bounds[6].max[0, 2] = 1

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(
        np.array([pose_at_first_node + [0] * n_qdot, pose_propulsion_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(
        np.array([pose_propulsion_start + [0] * n_qdot, pose_takeout_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(np.array(pose_at_first_node + [0] * n_qdot))
    x_init.add(
        np.array([pose_takeout_end + [0] * n_qdot, pose_salto_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_salto_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(
        np.array([pose_salto_end + [0] * n_qdot, pose_landing_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_landing_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * (bio_model[0].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[1].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[1].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[2].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[3].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[4].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[5].nb_tau - 3))

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

    # ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)

    # --- Show results --- #
    save_results_with_pickle(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    # sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
