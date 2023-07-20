"""
...

Phase 0: Preparation propulsion
- 3 contacts (TOE, HEEL)
- objectives functions: minimize torque, time

Phase 1: Propulsion
- 3 contacts (TOE, HEEL)
- objectives functions: minimize torque, time

Phase 2: Take-off phase
- zero contact
- objectives functions: minimize torque, time

Phase 3: Salto
- zero contact, holonomics constraints
- objectives functions: minimize torque, time

Phase 4: Transition Salto - landing
- zero contact
- objectives functions: minimize torque, time

"""
# --- Import package --- #

import numpy as np
import pickle
import bioviz
# import matplotlib.pyplot as plt
from bioptim import (
    Axis,
    BiorbdModel,
    Node,
    PhaseTransitionFcn,
    ConstraintFcn,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    DynamicsFunctions,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    PenaltyController,
    # HolonomicBiorbdModel,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
    QuadratureRule,
    PenaltyOption,
)
import matplotlib.pyplot as plt
from casadi import MX, vertcat, Function
from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_6phases, visualisation_movement




# --- Save results --- #
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


def save_results(sol, c3d_file_path, index_holonomics_constraints):
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

    if len(sol.ns) == 1:
        q = sol.states["u"]
        qdot = sol.states["udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == index_holonomics_constraints:
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
    # data["detailed_cost"] = sol.detailed_cost
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


def custom_phase_transition_pre(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

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
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

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


def compute_all_states(sol, bio_model: BiorbdModelCustomHolonomic, index_holonomics_constraints:int):
    """
    Compute all the states from the solution of the optimal control program

    Parameters
    ----------
    bio_model: HolonomicBiorbdModel
        The biorbd model
    sol:
        The solution of the optimal control program

    Returns
    -------

    """
    n = sol.states[index_holonomics_constraints]["q_u"].shape[1]
    nb_root = bio_model.nb_root
    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.ones((bio_model.nb_tau, n))
    tau_independent = [element - 3 for element in bio_model.independent_joint_index[3:]]
    tau_dependent = [element - 3 for element in bio_model.dependent_joint_index]

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index[3:]):
        tau[independent_joint_index] = sol.controls[index_holonomics_constraints]["tau"][tau_independent[i], :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index] = sol.controls[index_holonomics_constraints]["tau"][tau_dependent[i], :]

    # Partitioned forward dynamics
    q_u_sym = MX.sym("q_u_sym", bio_model.nb_independent_joints, 1)
    qdot_u_sym = MX.sym("qdot_u_sym", bio_model.nb_independent_joints, 1)
    tau_sym = MX.sym("tau_sym", bio_model.nb_tau, 1)
    partitioned_forward_dynamics_func = Function(
        "partitioned_forward_dynamics",
        [q_u_sym, qdot_u_sym, tau_sym],
        [bio_model.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym)],
    )
    # Lagrangian multipliers
    q_sym = MX.sym("q_sym", bio_model.nb_q, 1)
    qdot_sym = MX.sym("qdot_sym", bio_model.nb_q, 1)
    qddot_sym = MX.sym("qddot_sym", bio_model.nb_q, 1)
    compute_lambdas_func = Function(
        "compute_the_lagrangian_multipliers",
        [q_sym, qdot_sym, qddot_sym, tau_sym],
        [bio_model.compute_the_lagrangian_multipliers(q_sym, qdot_sym, qddot_sym, tau_sym)],
    )

    for i in range(n):
        q_v_i = bio_model.compute_v_from_u_explicit_numeric(sol.states[index_holonomics_constraints]["q_u"][:, i]).toarray()
        q[:, i] = bio_model.state_from_partition(sol.states[index_holonomics_constraints]["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        qdot[:, i] = bio_model.compute_qdot(q[:, i], sol.states[index_holonomics_constraints]["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = (
            partitioned_forward_dynamics_func(
                sol.states[index_holonomics_constraints]["q_u"][:, i],
                sol.states[index_holonomics_constraints]["qdot_u"][:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )
        qddot[:, i] = bio_model.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = (
            compute_lambdas_func(
                q[:, i],
                qdot[:, i],
                qddot[:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )

    return q, qdot, qddot, lambdas


def custom_minimize_q_udot(penalty: PenaltyOption, controller: PenaltyController):
    """
    Minimize the states variables.
    By default this function is quadratic, meaning that it minimizes towards the target.
    Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

    Parameters
    ----------
    penalty: PenaltyOption
        The actual penalty to declare
    controller: PenaltyController
        The penalty node elements
    """

    penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
    if (
            penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
            and penalty.integration_rule != QuadratureRule.TRAPEZOIDAL
    ):
        penalty.add_target_to_plot(controller=controller, combine_to="q_udot_states")
    penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

    # TODO: We should scale the target here!
    return controller.states["q_udot"].cx_start

def custom_minimize_q_u(penalty: PenaltyOption, controller: PenaltyController):
    """
    Minimize the states variables.
    By default this function is quadratic, meaning that it minimizes towards the target.
    Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

    Parameters
    ----------
    penalty: PenaltyOption
        The actual penalty to declare
    controller: PenaltyController
        The penalty node elements
    """

    penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
    if (
            penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
            and penalty.integration_rule != QuadratureRule.TRAPEZOIDAL
    ):
        penalty.add_target_to_plot(controller=controller, combine_to="q_u_states")
    penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

    # TODO: We should scale the target here!
    return controller.states["q_u"].cx_start

# --- Parameters --- #
movement = "Salto_close_loop"
version = 6
nb_phase = 6
index_holonomics_constraints = 3
independent_joint_index = [0, 1, 2, 5, 6, 7]
dependent_joint_index = [3, 4]

name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"
pickle_sol_init = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V13.pkl"
q_init, qdot_init, tau_init, time_init = get_created_data_from_pickle(pickle_sol_init)
q_init_holonomic = q_init[index_holonomics_constraints][independent_joint_index]
qdot_init_holonomic = qdot_init[index_holonomics_constraints][independent_joint_index]

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
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[3]),
                 BiorbdModel(biorbd_model_path[4]),
                 BiorbdModel(biorbd_model_path[5])
                 )

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0: Preparation propulsion
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.01, max_bound=0.6, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

    # Phase 1: Propulsion
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.01, max_bound=0.2, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1)

    # Phase 2: Take-off
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, min_bound=0.1, max_bound=0.4, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)

    # Phase 3: Salto
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, derivative=True, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.2, max_bound=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=3)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.ALL_SHOOTING, weight=-10000, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q_udot", weight=1, derivative=True, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q_u", weight=1, derivative=True, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q_udot", index=2, weight=-1, derivative=False,
    #                         quadratic=False, phase=3)
    # objective_functions.add(
    #     custom_minimize_q_udot,
    #     custom_type=ObjectiveFcn.Lagrange,
    #     quadratic=True,
    #     derivative=True,
    #     weight=1,
    #     phase=3,
    # )
    # objective_functions.add(
    #     custom_minimize_q_udot,
    #     custom_type=ObjectiveFcn.Lagrange,
    #     quadratic=False,
    #     derivative=False,
    #     weight=-1,
    #     phase=3,
    #     index=2,
    # )
    # objective_functions.add(
    #     custom_minimize_q_u,
    #     custom_type=ObjectiveFcn.Lagrange,
    #     quadratic=True,
    #     derivative=True,
    #     weight=1,
    #     phase=3,
    # )

    # Phase 4: Transition salto - take-off
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, min_bound=0.1, max_bound=0.4, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=4)

    # Phase 5 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=1000, phase=5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.4, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=5)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=1000, axes=Axis.Y, phase=5
    )

    variable_bimapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(
        bio_model[3].holonomic_torque_driven,
        dynamic_function=DynamicsFunctions.holonomic_torque_driven,
        mapping=variable_bimapping,
    )
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=5)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=2)
    # phase_transitions.add(custom_phase_transition_post, phase_pre_idx=3)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=4)

    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0)
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=1)
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=3)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Phase 0: Preparation propulsion
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

    # Phase 1: Propulsion
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

    # Create a holonomic constraint to create a double pendulum from two single pendulums
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[3],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=1,
    )
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model[3].set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=independent_joint_index,
        dependent_joint_index=dependent_joint_index,
    )

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

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]  # with elbow
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]  # with elbow
    tau_min = [i * 1 for i in tau_min_total]
    tau_max = [i * 1 for i in tau_max_total]
    # tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # Position solution
    # pose_at_first_node = [0.0188, 0.1368, -0.1091, 1.78, 0.5437, 0.191, -0.1452, 0.1821]  # Position of segment during first position
    # pose_propulsion_start = [-0.2347, -0.4555, -0.8645, 0.4820, 0.03, 2.5904, -2.2897, 0.5538]
    # pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.03, 0.5353, -0.8367, 0.1119]
    # pose_salto_start = [-0.3269, 0.6814, 0.9003, 0.35, 1.43, 2.3562, -2.3000, 0.6999]
    # pose_salto_end = [-0.8648, 1.3925, 3.7855, 0.35, 1.14, 2.3562, -2.3000, 0.6999]
    # pose_salto_start_CL = [-0.3269, 0.6814, 0.9003, 2.3562, -2.3000, 0.6999]
    # pose_salto_end_CL = [-0.8648, 1.3925, 3.7855, 2.3562, -2.3000, 0.6999]
    # pose_landing_start = [-0.9554, 0.1588, 5.8322, -0.4561, 0.03, 0.6704, -0.5305, 0.6546]
    # pose_landing_end = [-0.9461, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]

    pose_at_first_node = [0.0188, 0.1368, -0.1091, 1.78, 0.5437, 0.191, -0.1452,
                          0.25]  # Position of segment during first position
    pose_propulsion_start = [0.0195, -0.1714, -0.8568, -0.0782, 0.5437, 2.0522, -1.6462, 0.5296]
    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_salto_start_CL = [-0.6369, 1.0356, 1.5062, 2.1667, -1.9179, 0.0393]
    pose_salto_end_CL = [0.1987, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
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
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    # x_bounds[0:5]["q"].min[0, :] = -1
    # x_bounds[0:5]["q"].max[0, :] = 1

    # Phase 1: Propulsion
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, :] = -np.pi / 2
    x_bounds[1]["q"].max[2, :] = np.pi / 2
    x_bounds[1]["q"].min[0, :] = -1
    x_bounds[1]["q"].max[0, :] = 1

    # Phase 2: Take-off phase
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    # x_bounds[2]["q"].min[2, :] = -np.pi / 2
    # x_bounds[2]["q"].max[2, :] = 2 * np.pi
    # x_bounds[2]["q"].min[0, :] = -1
    # x_bounds[2]["q"].max[0, :] = 1
    # x_bounds[2]["q"].min[4, -1] = 1
    x_bounds[2]["q"].min[0, :] = -1
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 1:] = 0
    x_bounds[2]["q"].max[1, 1:] = 2.5
    x_bounds[2]["q"].min[2, 1] = -np.pi / 4
    x_bounds[2]["q"].max[2, 1] = np.pi / 2
    x_bounds[2]["q"].min[2, -1] = -np.pi / 2
    x_bounds[2]["q"].max[2, -1] = np.pi / 2
    x_bounds[2]["q"].min[4, -1] = 1

    x_bounds[2]["qdot"].min[0, :] = -5
    x_bounds[2]["qdot"].max[0, :] = 5
    x_bounds[2]["qdot"].min[1, :] = -2
    x_bounds[2]["qdot"].max[1, :] = 10
    x_bounds[2]["qdot"].min[2, :] = -5
    x_bounds[2]["qdot"].max[2, :] = 5

    # Phase 3: salto
    x_bounds.add("q_u", bounds=bio_model[3].bounds_from_ranges("q", mapping=variable_bimapping), phase=3)
    x_bounds.add("qdot_u", bounds=bio_model[3].bounds_from_ranges("qdot", mapping=variable_bimapping), phase=3)
    # x_bounds[3]["q_u"].min[2, 1] = -np.pi / 2
    # x_bounds[3]["q_u"].max[2, 1] = 2 * np.pi + 0.5
    # x_bounds[3]["q_u"].min[2, 2] = 2 * np.pi - 0.5
    # x_bounds[3]["q_u"].max[2, 2] = 2 * np.pi + 0.5
    # x_bounds[3]["q_u"].min[0, 2] = -1
    # x_bounds[3]["q_u"].max[0, 2] = 1

    x_bounds[3]["q_u"].min[0, :] = -2
    x_bounds[3]["q_u"].max[0, :] = 1
    x_bounds[3]["q_u"].min[1, 1:] = 0
    x_bounds[3]["q_u"].max[1, 1:] = 2.5
    x_bounds[3]["q_u"].min[2, 0] = 0
    x_bounds[3]["q_u"].max[2, 0] = np.pi / 2
    x_bounds[3]["q_u"].min[2, 1] = np.pi / 8
    x_bounds[3]["q_u"].max[2, 1] = 2 * np.pi
    x_bounds[3]["q_u"].min[2, 2] = 3/4 * np.pi
    x_bounds[3]["q_u"].max[2, 2] = 3/2 * np.pi
    x_bounds[3]["qdot_u"].min[0, :] = -5
    x_bounds[3]["qdot_u"].max[0, :] = 5
    x_bounds[3]["qdot_u"].min[1, :] = -2
    x_bounds[3]["qdot_u"].max[1, :] = 10
    x_bounds[3]["q_u"].max[3, :] = 2.6
    x_bounds[3]["q_u"].min[3, :] = 1.30

    # Phase 4: Take-off after salto
    x_bounds.add("q", bounds=bio_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=bio_model[4].bounds_from_ranges("qdot"), phase=4)
    # x_bounds[4]["q"].min[2, :] = 2 * np.pi - 0.5
    # x_bounds[4]["q"].max[2, :] = 2 * np.pi + 0.5
    # x_bounds[4]["q"].min[0, :] = -1
    # x_bounds[4]["q"].max[0, :] = 1

    x_bounds[4]["q"].min[0, :] = -2
    x_bounds[4]["q"].max[0, :] = 1
    x_bounds[4]["q"].min[1, 1:] = 0
    x_bounds[4]["q"].max[1, 1:] = 2.5
    x_bounds[4]["q"].min[2, :] = 3/4 * np.pi
    x_bounds[4]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[4]["qdot"].min[0, :] = -5
    x_bounds[4]["qdot"].max[0, :] = 5
    x_bounds[4]["qdot"].min[1, :] = -10
    x_bounds[4]["qdot"].max[1, :] = 10

    # Phase 5: landing
    x_bounds.add("q", bounds=bio_model[5].bounds_from_ranges("q"), phase=5)
    x_bounds.add("qdot", bounds=bio_model[5].bounds_from_ranges("qdot"), phase=5)
    x_bounds[5]["q"].min[2, :] = 2 * np.pi - 1.5
    x_bounds[5]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[5]["q"].min[0, 2] = -1
    x_bounds[5]["q"].max[0, 2] = 1
    x_bounds[5]["q"][:, 2] = pose_landing_end
    x_bounds[5]["qdot"][:, 2] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    # Phase 0 (prepa propulsion)
    # x_init.add("q", np.array([pose_at_first_node, pose_propulsion_start]).T, interpolation=InterpolationType.LINEAR,
    #            phase=0)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", q_init[0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", qdot_init[0], interpolation=InterpolationType.EACH_FRAME, phase=0)


    # Phase 1 (Propulsion)
    # x_init.add("q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", q_init[1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", qdot_init[1], interpolation=InterpolationType.EACH_FRAME, phase=1)


    # Phase 2 (take-off)
    # x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR,
    #            phase=2)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", q_init[2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot", qdot_init[2], interpolation=InterpolationType.EACH_FRAME, phase=2)


    # Phase 3 (salto)
    # x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
    #            phase=3)
    # x_init.add("q_udot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q_u", q_init_holonomic, interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("q_udot", qdot_init_holonomic, interpolation=InterpolationType.EACH_FRAME, phase=3)


    # Phase 4 (flight)
    # x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR,
    #            phase=4)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("q", q_init[4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("qdot", qdot_init[4], interpolation=InterpolationType.EACH_FRAME, phase=4)


    # Phase 5 (landing)
    # x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR,
    #            phase=5)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=5)
    x_init.add("q", q_init[5], interpolation=InterpolationType.EACH_FRAME, phase=5)
    x_init.add("qdot", qdot_init[5], interpolation=InterpolationType.EACH_FRAME, phase=5)


    # Define control path constraint
    u_bounds = BoundsList()
    for j in range(0, nb_phase):
        u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                     max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=j)

    u_init = InitialGuessList()
    for j in range(0, nb_phase):
        # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=j)
        u_init.add("tau", tau_init[j][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=j)


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
            assume_phase_dynamics=True,
            phase_transitions=phase_transitions,
            variable_mappings=dof_mapping,
        ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V2.bioMod"
    model_path_2contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_2contact,
                           model_path_1contact,
                           model_path,
                           model_path,
                           model_path,
                           model_path_2contact,
                           ),
        phase_time=(phase_time_init[0],
                    phase_time_init[1],
                    phase_time_init[2],
                    phase_time_init[3],
                    phase_time_init[4],
                    phase_time_init[5]),

        n_shooting=(n_shooting_init[0],
                    n_shooting_init[1],
                    n_shooting_init[2],
                    n_shooting_init[3],
                    n_shooting_init[4],
                    n_shooting_init[5]),
        # phase_time=(0.2, 0.1, 0.1, 0.4, 0.1, 0.2),
        # n_shooting=(20, 10, 10, 40, 10, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol = ocp.solve(solver)
    sol.graphs()
# --- Show results --- #
    save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl", index_holonomics_constraints=3)

    # sol.graphs(show_bounds=True)
    q, qdot, qddot, lambdas = compute_all_states(sol, bio_model[3], index_holonomics_constraints=index_holonomics_constraints)

    plt.plot(sol.time[index_holonomics_constraints], lambdas[0, :],
             label="y",
             marker="o",
             markersize=5,
             markerfacecolor="blue")
    plt.plot(sol.time[index_holonomics_constraints], lambdas[1, :],
             label="z",
             marker="o",
             markersize=5,
             markerfacecolor="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Lagrange multipliers (N)")
    plt.title("Lagrange multipliers of the holonomic constraint")
    plt.legend()
    plt.show()

    visualisation_closed_loop_6phases(bio_model, sol, model_path)


if __name__ == "__main__":
    main()