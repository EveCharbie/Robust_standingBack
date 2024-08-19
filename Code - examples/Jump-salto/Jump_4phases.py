"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): one contact (toe)
- Objective(s) function(s): maximize velocity of CoM and minimize time of flight

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact (in the air)
- Objective(s) function(s): maximize height of CoM and maximize time of flight

Phase 2: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact (in the air)
- Objective(s) function(s): maximize height of CoM and maximize time of flight

Phase 3: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): two contact
- Objective(s) function(s): minimize velocity CoM and minimize state qdot
"""

# --- Import package --- #
import numpy as np
import casadi as cas
import pickle
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    Axis,
    SolutionMerge,
    PenaltyController,
    PhaseTransitionFcn,
    DynamicsFunctions,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
)
from casadi import MX, vertcat

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
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    qddot =[]
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
            min_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].min)
            max_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].max)
            min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].min)
            max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].max)
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

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    #q_roots = controller.states["q_roots"].cx_start
    #q_joints = controller.states["q_joints"].cx_start
    #q = cas.vertcat(q_roots, q_joints)
    q = controller.states["q"].cx_start
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_index = controller.model.marker_index("Foot_Toe_marker")
    marker_pos = controller.model.markers(q)[marker_index]
    marker_pos_y = marker_pos[1]
    return marker_pos_y - CoM_pos_y

# --- Parameters --- #
movement = "Jump"
version = 10
nb_phase = 4
name_folder_model = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/Model"

# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
    )
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    # tau_min_total = [0, 0, 0, -162.7655, -69, -490.5938, -367.6643, -171.9903]
    # tau_max_total = [0, 0, 0, 162.7655, 69, 490.5938, 367.6643, 171.9903]
    tau_min = [i * 0.7 for i in tau_min_total]
    tau_max = [i * 0.7 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.01, max_bound=0.2, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.0001, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.0001, phase=0)

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=1)

    # Phase 2 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=2)

    # Phase 3 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.3, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=100, axes=Axis.Y,
                            phase=3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=3)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)

    # Constraints
    # Phase 0: Propulsion
    constraints = ConstraintList()
    # Phase 0 (Propulsion):
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.START,
        phase=0,
    )

    #constraints.add(
    #    ConstraintFcn.TRACK_MARKERS,
    #    marker_index="Shoulder_marker",
    #    axes=[Axis.X, Axis.Y, Axis.Z],
    #    max_bound=[0 + 0.10, 0.019 + 0.10, 1.149 + 0.10],
    #    min_bound=[0 - 0.10, 0.019 - 0.10, 1.149 - 0.10],
    #    node=Node.START,
    #    phase=0,
    #)

    #constraints.add(
    #    ConstraintFcn.TRACK_MARKERS,
    #    marker_index="KNEE_marker",
    #    axes=[Axis.X, Axis.Y, Axis.Z],
    #    max_bound=[0 + 0.10, 0.254 + 0.10, 0.413 + 0.10],
    #    min_bound=[0 - 0.10, 0.254 - 0.10, 0.413 - 0.10],
    #    node=Node.START,
    #    phase=0,
    #)

    constraints.add(
        CoM_over_toes,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    # Phase 3 (Landing):

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=3,
    )
    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Contraint position
    #pose_at_first_node = [-0.19, -0.43, -1.01, 0.0045, 2.5999, -2.2999, 0.6999]
    #pose_landing = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0]
    pose_propulsion_start = [0.0, -0.17, -0.9124, 0.0, 0.1936, 2.0082, -1.7997, 0.6472] # Plus de squat
    #pose_propulsion_start = [0, 0, -0.4535, -0.6596, 0.4259, 1.1334, -1.3841, 0.68]  # model bras en arriere
    pose_takeout_start = [0, 0, 0, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_tuck = [0, 1, 0.17, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393] #pose groupé
    #pose_takeout_end = [0, 1, 0, 2.5896, 0.51, 0, 0, 0.1119]
    pose_landing_start = [0, 0, 0.1930, 0.52, 0.95, 1.72, -0.81, 0.0]
    pose_landing_end = [0, 0, 0.1930, 3.1, 0.03, 0.0, 0.0, 0.0]

    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Propulsion
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"].min[2:7, 0] = np.array(pose_propulsion_start[2:7]) - 0.2
    x_bounds[0]["q"].max[2:7, 0] = np.array(pose_propulsion_start[2:7]) + 0.2
    #x_bounds[0]["q"].max[2, 0] = 0.5
    #x_bounds[0]["q"].max[5, 0] = 2
    #x_bounds[0]["q"].min[6, 0] = -2
    #x_bounds[0]["q"].max[6, 0] = -0.7
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot
    #x_bounds[0]["qdot"].max[5, :] = 0
    #x_bounds[0]["q"].min[2, 1:] = -np.pi
    #x_bounds[0]["q"].max[2, 1:] = np.pi
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["qdot"].min[3, :] = 0  # A commenter si marche pas
    #x_bounds[0]["q"].min[3, 2] = np.pi / 2

    # Phase 1: Flight
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[0, :] = -1
    x_bounds[1]["q"].max[0, :] = 1
    x_bounds[1]["q"].min[1, -1] = 0.5
    x_bounds[1]["q"].max[1, -1] = 3
    x_bounds[1]["q"].min[2:, -1] = np.array(pose_tuck[2:]) - 0.2


    # Phase 2: Second Flight
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -1
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 0] = 0.5
    x_bounds[2]["q"].max[1, 0] = 3

    # Phase 2: Landing
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].max[2:7, -1] = np.array(pose_landing_end[2:7]) + 0.2  # 0.5
    x_bounds[3]["q"].min[2:7, -1] = np.array(pose_landing_end[2:7]) - 0.2
    #x_bounds[3]["q"].min[5, 0] = pose_landing_start[5] - 1 #0.06
    #x_bounds[3]["q"].max[5, 0] = pose_landing_start[5] + 0.5
    #x_bounds[3]["q"].min[6, 0] = pose_landing_start[6] - 1
    #x_bounds[3]["q"].max[6, 0] = pose_landing_start[6] + 0.1
    x_bounds[3]["q"].min[0, :] = -1
    x_bounds[3]["q"].max[0, :] = 1
    #x_bounds[3]["q"].min[1, 0] = 0
    #x_bounds[3]["q"].max[1, 0] = 3
    #x_bounds[3]["q"].min[1, 1:] = -1
    #x_bounds[3]["q"].max[1, 1:] = 2.5
    #x_bounds[2]["qdot"][:, -1] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR,
               phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", np.array([pose_takeout_start, pose_tuck]).T, interpolation=InterpolationType.LINEAR,
               phase=1)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", np.array([pose_tuck, pose_landing_start]).T, interpolation=InterpolationType.LINEAR,
               phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=2)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=3)


    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[1].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[2].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[3].nb_tau - 3), phase=3)

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
        n_threads=32,
        # assume_phase_dynamics=True,
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V3.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V3.bioMod"

    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_1contact,
                           model_path,
                           model_path,
                           model_path_1contact),
        phase_time=(0.1, 0.2, 0.2, 0.3),
        n_shooting=(10, 20, 20, 30),
        min_bound=0.01,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    # ocp.add_plot_penalty()
    sol = ocp.solve(solver)
    sol.print_cost()

    # --- Show results --- #
    save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    sol.graphs(show_bounds=True, save_name=str(movement) + "_" + str(nb_phase) + "phases_V" + str(version))


if __name__ == "__main__":
    main()
