"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis and with the landing phase.
The simulation have 5 phases: propulsion, flight phase, tucked phase, preparation landing, landing.
We also want to see how well the transition between phases with and without holonomic constraints works.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 2: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 3: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 4: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time
"""

# --- Import package --- #
import numpy as np
import casadi as cas
import pickle
from bioptim import (
    BiorbdModel,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    SolutionMerge,
    PenaltyController,
    PhaseTransitionFcn,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
    Bounds,
    Node,
)
from casadi import MX, vertcat
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
from Save import get_created_data_from_pickle
from Salto_5phases_with_pelvis_landing import add_objectives, add_constraints, actuator_function, initialize_tau, add_x_bounds, add_u_bounds, save_results
from plot_actuators import Joint

# --- Save results --- #
def save_results_holonomic(sol, c3d_file_path, biomodel, index_holo):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """
    Q_sym = cas.MX.sym("Q_u", 6)
    Qdot_sym = cas.MX.sym("Qdot_u", 6)
    Tau_sym = cas.MX.sym("Tau", 8)
    lagrangian_multipliers_func = cas.Function(
        "Compute_lagrangian_multipliers",
        [Q_sym, Qdot_sym, Tau_sym],
        [biomodel[index_holo].compute_the_lagrangian_multipliers(Q_sym, Qdot_sym, Tau_sym)],
    )
    q_holo_func = cas.Function(
        "Compute_q_holo",
        [Q_sym],
        [biomodel[index_holo].state_from_partition(Q_sym, biomodel[index_holo].compute_v_from_u_explicit_symbolic(Q_sym))],
    )
    Bvu = biomodel[index_holo].coupling_matrix(q_holo_func(Q_sym))
    vdot = Bvu @ Qdot_sym
    qdot = biomodel[index_holo].state_from_partition(Qdot_sym, vdot)
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
    qddot =[]
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
                    lambdas[:, i_node] = np.reshape(lagrangian_multipliers_func(q_u[:, i_node], qdot_u[:, i_node], tau_this_time[:, i_node]), (2,))

                q.append(q_holo)
                qdot.append(qdot_holo)
                tau.append(controls[i]["tau"])
                time.append(list_time[i])
                min_bounds_q.append(sol.ocp.nlp[i].x_bounds['q_u'].min)
                max_bounds_q.append(sol.ocp.nlp[i].x_bounds['q_u'].max)
                min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot_u'].min)
                max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot_u'].max)
                min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
                max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)
            else:
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

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def custom_phase_transition_pre(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    states_pre = controllers[0].states.cx

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    states_post = vertcat(q_post, qdot_post)

    return states_pre - states_post


def custom_phase_transition_post(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    nb_independent = controllers[0].model.nb_independent_joints
    u_pre = controllers[0].states.cx[:nb_independent]
    udot_pre = controllers[0].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    q_pre = controllers[0].model.state_from_partition(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.state_from_partition(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)
    states_post = controllers[1].states.cx

    return states_pre - states_post


def custom_contraint_lambdas_normal(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:

    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    return lagrange_0

def custom_contraint_lambdas_cisaillement(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    """
    lagrange_1**2 < lagrange_0**2
    """
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)
    lagrange_0 = lambdas[0]
    lagrange_1 = lambdas[1]

    return lagrange_0**2 - lagrange_1**2


def custom_contraint_lambdas_cisaillement_min_bound(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    """
    lagrange_1 < lagrange_0
    """
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    # Contrainte lagrange_1 (min_bound = L_1/L_0 = -0.2, max_bound = L_1/L_0 = 0.2)
    lagrange_1 = lambdas[1]

    return -(lagrange_1 - lagrange_0)


def custom_contraint_lambdas_cisaillement_max_bound(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    """
    0.01*lagrange_0 < lagrange_1
    """
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    # Contrainte lagrange_1 (min_bound = L_1/L_0 = -0.2, max_bound = L_1/L_0 = 0.2)
    lagrange_1 = lambdas[1]

    return -(lagrange_1 - 0.01 * lagrange_0)

def minimize_actuator_torques_CL(controller: PenaltyController, actuators) -> cas.MX:

    nb_independent = controller.model.nb_independent_joints
    u = controller.states.cx[:nb_independent]
    v = controller.model.compute_v_from_u_explicit_symbolic(u)
    q = controller.model.state_from_partition(u, v)

    tau = controller.controls["tau"].cx_start
    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i+3]),
            actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i+3]))
        out += (tau[i] / current_max_tau)**2
    return cas.sum1(out)


# --- Parameters --- #
movement = "Salto_close_loop_landing"
version = "Eve14"
nb_phase = 5
name_folder_model = "../Model"
# pickle_sol_init = "init/Salto_close_loop_landing_5phases_VEve12.pkl"
# pickle_sol_init = "/home/mickaelbegon/Documents/Anais/Results_simu/Salto_5phases_VEve3.pkl"
pickle_sol_init = "init/Jump_4phases_V22.pkl"
sol_salto = get_created_data_from_pickle(pickle_sol_init)

# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[2]),
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
    variable_bimapping = BiMappingList()
    dof_mapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions = add_objectives(objective_functions, actuators)
    objective_functions.add(
        minimize_actuator_torques_CL,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=2,
    )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=4)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=1)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    constraints = add_constraints(constraints)
    holonomic_constraints = HolonomicConstraintsList()

    # Phase 2 (Tucked phase):
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[2],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
    )

    bio_model[2].set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )

    # "relaxed friction cone"
    constraints.add(
        custom_contraint_lambdas_cisaillement,
        node=Node.ALL_SHOOTING,
        bio_model=bio_model[2],
        max_bound=np.inf,
        min_bound=0,
        phase=2,
    )
    # The model can only pull on the legs, not push
    constraints.add(
        custom_contraint_lambdas_normal,
        node=Node.ALL_SHOOTING,
        bio_model=bio_model[2],
        max_bound=-0.1,
        min_bound=-np.inf,
        phase=2,
    )

    # --- Bounds ---#
    x_bounds = BoundsList()
    q_bounds, qdot_bounds = add_x_bounds(bio_model)
    for i_phase in range(len(bio_model)):
        if i_phase == 2:
            qu_bounds = Bounds("q_u", min_bound=variable_bimapping["q"].to_first.map(q_bounds[i_phase].min),
                               max_bound=variable_bimapping["q"].to_first.map(q_bounds[i_phase].max))
            qdotu_bounds = Bounds("qdot_u", min_bound=variable_bimapping["qdot"].to_first.map(qdot_bounds[i_phase].min),
                                  max_bound=variable_bimapping["qdot"].to_first.map(qdot_bounds[i_phase].max))
            x_bounds.add("q_u", bounds=qu_bounds, phase=i_phase)
            x_bounds.add("qdot_u", bounds=qdotu_bounds, phase=i_phase)
        else:
            x_bounds.add("q", bounds=q_bounds[i_phase], phase=i_phase)
            x_bounds.add("qdot", bounds=qdot_bounds[i_phase], phase=i_phase)

    # Initial guess
    pose_salto_start = [0.135, 0.455, 1.285, 0.481, 1.818, 2.6, -1.658, 0.692]
    pose_salto_end = [0.107, 0.797, 2.892, 0.216, 1.954, 2.599, -2.058, 0.224]

    x_init = InitialGuessList()
    # Init straight jump
    x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    q_2_linear = variable_bimapping["q"].to_first.map(np.linspace(pose_salto_start, pose_salto_end, n_shooting[2]+1).T)
    x_init.add("q_u", q_2_linear, interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot_u", [0] * 6, interpolation=InterpolationType.CONSTANT, phase=2)
    x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Init CL salto
    # x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # x_init.add("q_u", sol_salto["q_u"], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("qdot_u", sol_salto["qdot_u"], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("q", sol_salto["q"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("qdot", sol_salto["qdot"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds = add_u_bounds(u_bounds, tau_min, tau_max)

    u_init = InitialGuessList()
    # Init straight jump
    u_init.add("tau", sol_salto["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol_salto["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    u_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Init CL salto
    # u_init.add("tau", sol_salto["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    # u_init.add("tau", sol_salto["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    # u_init.add("tau", sol_salto["tau"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # u_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # u_init.add("tau", sol_salto["tau"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)

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
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V3.bioMod"
    #model_path_2contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V3.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V3.bioMod"

    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_1contact,
                           model_path,
                           model_path,
                           model_path,
                           model_path_1contact),
        phase_time=(0.2, 0.2, 0.3, 0.3, 0.3),
        n_shooting=(20, 20, 30, 30, 30),
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")#show_online_optim=True,
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)
    # solver.set_nlp_scaling_method("none")  # Doesn't work
    #ocp.add_plot_penalty()
    sol = ocp.solve(solver)
    sol.print_cost()


# --- Save results --- #
    save_results_holonomic(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + version + ".pkl", bio_model, 2)
    # save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + version + ".pkl")
    sol.graphs(show_bounds=True, save_name=str(movement) + "_" + str(nb_phase) + "phases_V" + str(version))
    sol.animate()

if __name__ == "__main__":
    main()