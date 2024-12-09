"""
TODO: Add description here
x = [q, qdot, tau], u = [taudot]
"""

# --- Import package --- #
import os
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
    MagnitudeType,
    MultiStart,
    OnlineOptim,
)
from Save import get_created_data_from_pickle
from plot_actuators import Joint, actuator_function

# --- Save results --- #
def save_results(sol,
                 *combinatorial_parameters,
                 **extra_parameters,):

    biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]

    folder_path = f"{save_folder}/{str(movement)}_{str(nb_phase)}phases_V{version}"
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
        min_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].min)
        max_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].max)
        min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].min)
        max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].max)
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

    return


def CoM_over_toes(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_index = controller.model.marker_index("Foot_Toe_marker")
    marker_pos = controller.model.markers(q)[marker_index]
    marker_pos_y = marker_pos[1]
    constraint = marker_pos_y - CoM_pos_y
    return constraint

def minimize_actuator_torques(controller: PenaltyController, actuators) -> cas.MX:
    q = controller.states["q"].cx_start
    tau = controller.states["tau"].cx_start
    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i+3]),
            actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i+3]))
        out += (tau[i] / current_max_tau)**2
    return cas.sum1(out)

def add_objectives(objective_functions, actuators):

    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.1, max_bound=0.4, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=0)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.01,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES_END_OF_INTERVAL,
        node=Node.PENULTIMATE,
        weight=0.01,
        contact_index=1,
        quadratic=True,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES_END_OF_INTERVAL,
        node=Node.PENULTIMATE,
        weight=0.01,
        contact_index=0,
        quadratic=True,
        phase=0,
    )

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=1)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=1,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=1)

    # Phase 2 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, min_bound=0.1, max_bound=0.4, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=2)

    # Phase 3 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=3)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=3,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=3)

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=100, phase=4)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=4)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.2, max_bound=1, phase=4)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.01,
        phase=4,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=4)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=100, axes=Axis.Y,
                            phase=4)

    return objective_functions


def add_constraints(constraints):

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

    #  PIERRE THIS IS MISSING RIGHT ?
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        marker_index="Foot_Toe_marker",
        node=Node.START,
        phase=0,
    )

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

    constraints.add(
       ConstraintFcn.TRACK_CONTACT_FORCES_END_OF_INTERVAL,
       node=Node.PENULTIMATE,
       contact_index=1,
       quadratic=True,
       phase=0,
    )

    # Phase 4 (Landing):
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.END,
        phase=4,
    )

    #  PIERRE THIS IS MISSING RIGHT ?
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        marker_index="Foot_Toe_marker",
        node=Node.START,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=4,
    )
    return constraints

def initialize_tau():
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.7 for i in tau_min_total]
    tau_max = [i * 0.7 for i in tau_max_total]
    tau_init = 0
    return tau_min, tau_max, tau_init

def add_x_bounds(bio_model):

    n_q = bio_model[0].nb_q
    n_qdot = n_q

    q_bounds = [bio_model[0].bounds_from_ranges("q") for i in range(len(bio_model))]
    qdot_bounds = [bio_model[0].bounds_from_ranges("qdot") for i in range(len(bio_model))]

    pose_propulsion_start = [-0.2343, -0.2177, -0.3274, 0.2999, 0.4935, 1.7082, -1.9999, 0.1692]
    pose_landing_start = [0.013, 0.088, 5.804, -0.305, 8.258444956622276e-06, 1.014, -0.97, 0.006]
    pose_landing_end = [0.053, 0.091, 6.08, 2.9, -0.17, 0.092, 0.17, 0.20]

    # Phase 0: Propulsion
    q_bounds[0].min[2:7, 0] = np.array(pose_propulsion_start[2:7]) - 0.3
    q_bounds[0].max[2:7, 0] = np.array(pose_propulsion_start[2:7]) + 0.3
    q_bounds[0].max[2, 0] = 0.5
    q_bounds[0].max[5, 0] = 2
    q_bounds[0].min[6, 0] = -2
    q_bounds[0].max[6, 0] = -0.7
    qdot_bounds[0][:, 0] = [0] * n_qdot
    qdot_bounds[0].max[5, :] = 0
    q_bounds[0].min[2, 1:] = -np.pi
    q_bounds[0].max[2, 1:] = np.pi
    q_bounds[0].min[0, :] = -1
    q_bounds[0].max[0, :] = 1
    q_bounds[0].min[1, :] = -1
    q_bounds[0].max[1, :] = 2
    qdot_bounds[0].min[3, :] = 0   # A commenter si marche pas
    q_bounds[0].min[3, 2] = np.pi/2

    # Phase 1: Flight
    q_bounds[1].min[0, :] = -1
    q_bounds[1].max[0, :] = 1
    q_bounds[1].min[1, :] = -1
    q_bounds[1].max[1, :] = 2
    q_bounds[1].min[2, 0] = -np.pi
    q_bounds[1].max[2, 0] = np.pi * 1.5
    q_bounds[1].min[2, 1] = -np.pi
    q_bounds[1].max[2, 1] = np.pi * 1.5
    q_bounds[1].min[2, -1] = -np.pi
    q_bounds[1].max[2, -1] = np.pi * 1.5

    # Phase 2: Tucked phase
    q_bounds[2].min[0, :] = - 1
    q_bounds[2].max[0, :] = 1
    q_bounds[2].min[1, :] = -1
    q_bounds[2].max[1, :] = 2
    q_bounds[2].min[2, 0] = -np.pi
    q_bounds[2].max[2, 0] = np.pi * 1.5
    q_bounds[2].min[2, 1] = -np.pi
    q_bounds[2].max[2, 1] = 2 * np.pi
    q_bounds[2].min[2, 2] = 3/4 * np.pi
    q_bounds[2].max[2, 2] = 3/2 * np.pi
    q_bounds[2].max[5, :-1] = 2.6
    q_bounds[2].min[5, :-1] = 1.96
    q_bounds[2].max[6, :-1] = -1.5
    q_bounds[2].min[6, :-1] = -2.3

    # Phase 3: Preparation landing
    q_bounds[3].min[0, :] = -1
    q_bounds[3].max[0, :] = 1
    q_bounds[3].min[1, 1:] = -1
    q_bounds[3].max[1, 1:] = 2
    q_bounds[3].min[2, :] = 3/4 * np.pi
    q_bounds[3].max[2, :] = 2 * np.pi + 0.5
    q_bounds[3].min[5, -1] = pose_landing_start[5] - 1 #0.06
    q_bounds[3].max[5, -1] = pose_landing_start[5] + 0.5
    q_bounds[3].min[6, -1] = pose_landing_start[6] - 1
    q_bounds[3].max[6, -1] = pose_landing_start[6] + 0.1

    # Phase 3: Landing
    q_bounds[4].min[5, 0] = pose_landing_start[5] - 1 #0.06
    q_bounds[4].max[5, 0] = pose_landing_start[5] + 0.5
    q_bounds[4].min[6, 0] = pose_landing_start[6] - 1
    q_bounds[4].max[6, 0] = pose_landing_start[6] + 0.1
    q_bounds[4].min[0, :] = -1
    q_bounds[4].max[0, :] = 1
    q_bounds[4].min[1, :] = -1
    q_bounds[4].max[1, :] = 2
    q_bounds[4].min[2, 0] = 2/4 * np.pi
    q_bounds[4].max[2, 0] = 2 * np.pi + 1.66
    q_bounds[4].min[2, 1] = 2/4 * np.pi
    q_bounds[4].max[2, 1] = 2 * np.pi + 1.66
    q_bounds[4].max[:, -1] = np.array(pose_landing_end) + 0.2 #0.5
    q_bounds[4].min[:, -1] = np.array(pose_landing_end) - 0.2

    return q_bounds, qdot_bounds


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed=0):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 BiorbdModel(biorbd_model_path[3]),
                 BiorbdModel(biorbd_model_path[4]),
                 )

    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Actuators parameters
    actuators = {"Shoulders": Joint(tau_max_plus=112.8107 * 2,
                                    theta_opt_plus=-41.0307 * np.pi / 180,
                                    r_plus=109.6679 * np.pi / 180,
                                    tau_max_minus=162.7655 * 2,
                                    theta_opt_minus=-101.6627 * np.pi / 180,
                                    r_minus=103.9095 * np.pi / 180,
                                    min_q=-0.7,
                                    max_q=3.1),
                 "Elbows": Joint(tau_max_plus=80 * 2,
                                 theta_opt_plus=np.pi / 2 - 0.1,
                                 r_plus=40 * np.pi / 180,
                                 tau_max_minus=50 * 2,
                                 theta_opt_minus=np.pi / 2 - 0.1,
                                 r_minus=70 * np.pi / 180,
                                 min_q=0,
                                 max_q=2.09),
                 # this one was not measured, I just tried to fit https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
                 "Hips": Joint(tau_max_plus=220.3831 * 2,
                               theta_opt_plus=25.6939 * np.pi / 180,
                               r_plus=56.4021 * np.pi / 180,
                               tau_max_minus=490.5938 * 2,
                               theta_opt_minus=72.5836 * np.pi / 180,
                               r_minus=48.6999 * np.pi / 180,
                               min_q=-0.4,
                               max_q=2.6),
                 "Knees": Joint(tau_max_plus=367.6643 * 2,
                                theta_opt_plus=-61.7303 * np.pi / 180,
                                r_plus=31.7218 * np.pi / 180,
                                tau_max_minus=177.9694 * 2,
                                theta_opt_minus=-33.2908 * np.pi / 180,
                                r_minus=57.0370 * np.pi / 180,
                                min_q=-2.3,
                                max_q=0.02),
                 "Ankles": Joint(tau_max_plus=153.8230 * 2,
                                 theta_opt_plus=0.7442 * np.pi / 180,
                                 r_plus=58.9832 * np.pi / 180,
                                 tau_max_minus=171.9903 * 2,
                                 theta_opt_minus=12.6824 * np.pi / 180,
                                 r_minus=21.8717 * np.pi / 180,
                                 min_q=-0.7,
                                 max_q=0.7)
                 }

    tau_min, tau_max, tau_init = initialize_tau()
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])
    dof_mapping.add("taudot", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions = add_objectives(objective_functions, actuators)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.01,
        phase=2,
    )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=4)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.TAKEOFF, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # --- Constraints ---#
    constraints = ConstraintList()
    constraints = add_constraints(constraints)

    # Path constraint
    pose_salto_start = [0.135, 0.455, 1.285, 0.481, 1.818, 2.6, -1.658, 0.692]
    pose_salto_end = [0.107, 0.797, 2.892, 0.216, 1.954, 2.599, -2.058, 0.224]
    pose_landing_start = [0.013, 0.088, 5.804, -0.305, 8.258444956622276e-06, 1.014, -0.97, 0.006]

    # --- Bounds ---#
    x_bounds = BoundsList()
    q_bounds, qdot_bounds = add_x_bounds(bio_model)
    for i_phase in range(len(bio_model)):
        x_bounds.add("q", bounds=q_bounds[i_phase], phase=i_phase)
        x_bounds.add("qdot", bounds=qdot_bounds[i_phase], phase=i_phase)
        x_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                     max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=i_phase)

    # Initial guess
    x_init = InitialGuessList()
    # Initial guess from Jump
    x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q", np.array([pose_salto_start, pose_salto_end]).T,
               interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T,
               interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("tau", np.hstack((sol_salto["tau"][0], np.zeros((5, 1)))), interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("tau", np.hstack((sol_salto["tau"][1], np.zeros((5, 1)))), interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    x_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    x_init.add("tau", np.hstack((sol_salto["tau"][3], np.zeros((5, 1)))), interpolation=InterpolationType.EACH_FRAME, phase=4)

    # # Initial guess from somersault
    # x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("q", sol_salto["q"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("qdot", sol_salto["qdot"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("q", sol_salto["q"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("qdot", sol_salto["qdot"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("tau", sol_salto["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("tau", sol_salto["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("tau", sol_salto["tau"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("tau", sol_salto["tau"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for i_phase in range(len(bio_model)):
        u_bounds.add("taudot", min_bound=[-10000]*5, max_bound=[10000]*5, phase=i_phase)
        u_init.add("taudot", [0]*5, phase=i_phase)


    if WITH_MULTI_START:
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.2,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] + 1 for i in range(len(n_shooting))],
            seed=seed,
        )
        u_init.add_noise(
            bounds=u_bounds,
            magnitude=0.01,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] for i in range(len(n_shooting))],
            seed=seed,
        )

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
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
    )


def prepare_multi_start(
    combinatorial_parameters: dict,
    save_folder: str = None,
    n_pools: int = 10,
    solver: Solver = None,
):
    """
    The initialization of the multi-start
    """

    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results, {"save_folder": save_folder}),
        should_solve_callback=(should_solve, {"save_folder": save_folder}),
        n_pools=n_pools,
        solver=solver,
    )


def should_solve(*combinatorial_parameters, **extra_parameters):
    return True



# --- Parameters --- #
movement = "Salto"
version = "Eve_taudot1"
nb_phase = 5
name_folder_model = "../Model"
pickle_sol_init = "/home/mickaelbegon/Documents/Anais/Results_simu/Jump_4phases_V22.pkl"
# pickle_sol_init = "/home/mickaelbegon/Documents/Anais/Results_simu/Salto_close_loop_landing_5phases_VEve12.pkl"
sol_salto = get_created_data_from_pickle(pickle_sol_init)



# --- Load model --- #
def main():

    WITH_MULTI_START = False

    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V3.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V3.bioMod"

    # Solver options
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")#show_online_optim=True,
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)

    biorbd_model_path = [(model_path_1contact,
                         model_path,
                         model_path,
                         model_path,
                         model_path_1contact)]
    phase_time = [(0.2, 0.2, 0.3, 0.3, 0.3)]
    n_shooting = [(20, 20, 30, 30, 30)]

    seed = list(range(20))
    combinatorial_parameters = {
        "bio_model_path": biorbd_model_path,
        "phase_time": phase_time,
        "n_shooting": n_shooting,
        "WITH_MULTI_START": [True],
        "seed": seed,
    }


    if WITH_MULTI_START:
        save_folder = "./solutions"
        multi_start = prepare_multi_start(
            combinatorial_parameters=combinatorial_parameters,
            save_folder=save_folder,
            solver=solver,
            # n_pools=1,
        )

        multi_start.solve()
    else:
        ocp = prepare_ocp(biorbd_model_path[0], phase_time[0], n_shooting[0], WITH_MULTI_START=False)
        ocp.add_plot_penalty()

        solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")  # online_optim=OnlineOptim.DEFAULT
        solver.set_maximum_iterations(50000)
        solver.set_bound_frac(1e-8)
        solver.set_bound_push(1e-8)
        solver.set_tol(1e-6)

        sol = ocp.solve(solver)
        sol.print_cost()

        # --- Save results --- #
        save_results(sol, combinatorial_parameters)
        sol.graphs(show_bounds=True, save_name=str(movement) + "_" + str(nb_phase) + "phases_V" + version)
        # sol.animate()


if __name__ == "__main__":
    main()
