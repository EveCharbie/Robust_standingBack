"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis, the propulsion phase and the landing phase.
The simulation have 6 phases: preparation propulsion, propulsion, flight phase, tucked phase, preparation landing, landing.
We also want to see how well the transition between phases with and without holonomic constraints works.

Phase 0: Preparayion propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 2: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 3: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 4: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 5: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

"""
# --- Import package --- #

import numpy as np
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
    PenaltyController,
    PhaseTransitionFcn,
    DynamicsFunctions,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
    MultinodeObjectiveList,
    MultinodeConstraintFcn,
    MultinodeObjectiveFcn,
)
from casadi import MX, vertcat
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_6phases
from Save import get_created_data_from_pickle
from casadi import MX, sum1, sum2
# --- Save results --- #


def custom_multinode_states_equality(
        controllers: list[PenaltyController, ...],
):
    """
    The most common continuity function, that is state before equals state after

    Parameters
    ----------
    controllers: list
        The penalty node elements

    Returns
    -------
    The difference between the state after and before
    """
    ctrl_0 = controllers[0].states["qdot"].cx_end
    ctrl_1 = controllers[1].states["qdot"].cx_start

    dt = controllers[0].tf / controllers[0].ns
    out = 0
    for i, ctrl in enumerate(controllers):
        out += sum1(ctrl.states["qdot"].cx_start ** 2) * dt
        out += ctrl_0 - ctrl_1

    return out


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

    if len(sol.ns) == 1:
        q = sol.states["q_u"]
        qdot = sol.states["q_udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 3:
                q.append(sol.states[i]["q_u"])
                qdot.append(sol.states[i]["qdot_u"])
                # states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])
            else:
                q.append(sol.states[i]["q"])
                qdot.append(sol.states[i]["qdot"])
                # states_all.append(sol.states[i]["all"])
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


# --- Parameters --- #
movement = "Salto_close_loop_landing"
version = 20
nb_phase = 6
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"
pickle_sol_init = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_close_loop_landing_4phases_V13.pkl"
sol = get_created_data_from_pickle(pickle_sol_init)


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[3]),
                 BiorbdModel(biorbd_model_path[4]),
                 BiorbdModel(biorbd_model_path[5]),
                 )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.8 for i in tau_min_total]
    tau_max = [i * 0.8 for i in tau_max_total]
    tau_init = 0
    variable_bimapping = BiMappingList()
    dof_mapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives = MultinodeObjectiveList()

    # Phase 0: Preparation propulsion
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.01, max_bound=0.6, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.000001, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.000001, phase=0)

    # Phase 1: Propulsion
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.01, max_bound=0.2, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)

    # Phase 2: Flight
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=2)

    # Phase 3: Tucked phase
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.4, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=3)

    # Phase 4: Preparation landing
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=4)

    # Phase 5: Landing
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=1000, phase=5)  # TODO: Changer
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.3, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=5)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=1000, axes=Axis.Y,
                            phase=5)
    # multinode_objectives.add(
    #     custom_multinode_states_equality,
    #     nodes_phase=(4, 5),
    #     nodes=(Node.END, Node.START),
    #     key="qdot",
    #     weight=10000,
    # )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(
        bio_model[3].holonomic_torque_driven,
        dynamic_function=DynamicsFunctions.holonomic_torque_driven,
        mapping=variable_bimapping,
        phase=3,
    )
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=5)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=2)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=3)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=4)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    holonomic_constraints = HolonomicConstraintsList()

    # Phase 0: Preparation propulsion
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.END,
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
        node=Node.END,
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
        phase=0,
    )

    # Phase 3: Tucked phase
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[3],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
    )
    # Made up constraints

    bio_model[3].set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )
    # Phase 5: Landing
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
    pose_at_first_node = [0.0188, 0.1368, -0.1091, 1.78, 0.5437, 0.191, -0.1452,
                          0.1821]  # Position of segment during first position
    pose_propulsion_start = [-0.2347217373715483, -0.45549996131551357, -0.8645258574574489, 0.4820766547674885, 0.03,
                             2.590467089448695, -2.289747592408045, 0.5538056491954265]
    pose_takeout_start = [-0.2777672842694191, 0.03995514292843797, 0.1930477703559439, 2.589642304908377, 0.03,
                          0.5353536016159908, -0.8367077461678971, 0.11196901833050495]
    pose_salto_start = [-0.3269534844623969, 0.681422172573302, 0.9003344030624946, 0.35, 1.43, 2.3561945135532367,
                        -2.300000008273391, 0.6999999941919349]
    pose_salto_end = [-0.8648803377623905, 1.3925287774995057, 3.785530485157555, 0.35, 1.14, 2.3561945105754827,
                      -2.300000018314619, 0.6999999322366998]
    pose_salto_start_CL = [-0.3269534844623969, 0.681422172573302, 0.9003344030624946, 2.3561945135532367,
                           -2.300000008273391, 0.6999999941919349]
    pose_salto_end_CL = [-0.8648803377623905, 1.3925287774995057, 3.785530485157555, 2.3561945105754827,
                         -2.300000018314619, 0.6999999322366998]
    pose_landing_start = [-0.9554004763233065, 0.15886445602166693, 5.832254254152056, -0.45610833795726297, 0.03,
                          0.85, -1.39, 0.654641794221728]
    pose_landing_end = [-0.9461201943294933, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[3].nb_independent_joints

    # Phase 0: Pareparation propulsion
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_at_first_node
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot  # impose the first position
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 0.5

    # Phase 1: Propulsion
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, :] = -np.pi / 2
    x_bounds[1]["q"].max[2, :] = np.pi / 2
    x_bounds[1]["q"].min[0, :] = -1
    x_bounds[1]["q"].max[0, :] = 0.5

    # Phase 2: Flight
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -1
    x_bounds[2]["q"].max[0, :] = 0.5
    x_bounds[2]["q"].min[1, :] = 0
    x_bounds[2]["q"].max[1, :] = 2.5
    x_bounds[2]["q"].min[2, 0] = -np.pi / 2
    x_bounds[2]["q"].max[2, 0] = np.pi / 2
    # x_bounds[0]["q"].min[2, 0] = -np.pi / 4
    # x_bounds[0]["q"].max[2, 0] = np.pi / 4
    x_bounds[2]["q"].min[2, 1] = -np.pi / 4
    x_bounds[2]["q"].max[2, 1] = np.pi / 2
    x_bounds[2]["q"].min[2, -1] = np.pi / 2
    x_bounds[2]["q"].max[2, -1] = np.pi
    x_bounds[2]["q"].min[4, -1] = 1
    # x_bounds[1]["qdot"].min[0, :] = -5
    # x_bounds[1]["qdot"].max[0, :] = 5
    # x_bounds[1]["qdot"].min[1, :] = -10
    # x_bounds[1]["qdot"].max[1, :] = 10
    # x_bounds[1]["qdot"].min[2, :] = -5
    # x_bounds[1]["qdot"].max[2, :] = 5
    # x_bounds[0]["q"].max[4, -1] = -1
    # x_bounds[0]["q"].min[4, -1] = -2.3


    # Phase 3: Tucked phase
    x_bounds.add("q_u", bounds=bio_model[3].bounds_from_ranges("q", mapping=variable_bimapping), phase=3)
    x_bounds.add("qdot_u", bounds=bio_model[3].bounds_from_ranges("qdot", mapping=variable_bimapping), phase=3)
    x_bounds[3]["q_u"].min[0, :] = -2
    x_bounds[3]["q_u"].max[0, :] = 0.5
    x_bounds[3]["q_u"].min[1, 1:] = 0
    x_bounds[3]["q_u"].max[1, 1:] = 2.5
    x_bounds[3]["q_u"].min[2, 0] = 0
    x_bounds[3]["q_u"].max[2, 0] = np.pi / 2
    x_bounds[3]["q_u"].min[2, 1] = np.pi / 8
    x_bounds[3]["q_u"].max[2, 1] = 2 * np.pi
    x_bounds[3]["q_u"].min[2, 2] = 3/4 * np.pi
    x_bounds[3]["q_u"].max[2, 2] = 3/2 * np.pi
    # x_bounds[2]["qdot_u"].min[0, :] = -5
    # x_bounds[2]["qdot_u"].max[0, :] = 5
    # x_bounds[2]["qdot_u"].min[1, :] = -2
    # x_bounds[2]["qdot_u"].max[1, :] = 10
    x_bounds[3]["q_u"].max[3, :-1] = 2.6
    x_bounds[3]["q_u"].min[3, :-1] = 1.96
    x_bounds[3]["q_u"].max[4, :-1] = -1.72
    x_bounds[3]["q_u"].min[4, :-1] = -2.3

    # Phase 4: Preparation landing
    x_bounds.add("q", bounds=bio_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=bio_model[4].bounds_from_ranges("qdot"), phase=4)
    x_bounds[4]["q"].min[0, :] = -2
    x_bounds[4]["q"].max[0, :] = 0.5
    x_bounds[4]["q"].min[1, 1:] = 0
    x_bounds[4]["q"].max[1, 1:] = 2.5
    x_bounds[4]["q"].min[2, :] = 3/4 * np.pi
    x_bounds[4]["q"].max[2, :] = 2 * np.pi + 0.5
    #x_bounds[2]["q"].max[5, :] = 0.2
    # x_bounds[3]["qdot"].min[0, :] = -5
    # x_bounds[3]["qdot"].max[0, :] = 5
    # x_bounds[3]["qdot"].min[1, :] = -10
    # x_bounds[3]["qdot"].max[1, :] = 10

    # Phase 5: Landing
    x_bounds.add("q", bounds=bio_model[5].bounds_from_ranges("q"), phase=5)
    x_bounds.add("qdot", bounds=bio_model[5].bounds_from_ranges("qdot"), phase=5)
    x_bounds[5]["q"][[5, 6, 7], 0] = pose_landing_start[5:]
    x_bounds[5]["q"].min[0, :] = -2
    x_bounds[5]["q"].max[0, :] = 0.5
    x_bounds[5]["q"].min[1, 0] = 0
    x_bounds[5]["q"].max[1, 0] = 2.5
    x_bounds[5]["q"].min[1, 1:] = -1
    x_bounds[5]["q"].max[1, 1:] = 2.5
    x_bounds[5]["q"].min[2, 0] = 3/4 * np.pi
    x_bounds[5]["q"].max[2, 0] = 2 * np.pi + 0.5
    x_bounds[5]["q"].min[2, 1:] = 2 * np.pi - 0.5
    x_bounds[5]["q"].max[2, 1:] = 2 * np.pi + 0.5
    # x_bounds[3]["q"].max[7, :] = 0.3
    # x_bounds[3]["q"].min[7, :] = -0.3
    x_bounds[5]["q"][:, -1] = pose_landing_end
    # x_bounds[3]["q"].max[:, -1] = np.array(pose_landing_end) + 0.5
    # x_bounds[3]["q"].min[:, -1] = np.array(pose_landing_end) - 0.5

    # x_bounds[3]["q"].min[2, :] = 2 * np.pi - 1.5
    # x_bounds[3]["q"].max[2, :] = 2 * np.pi + 0.5
    # x_bounds[3]["q"].min[0, 2] = -1
    # x_bounds[3]["q"].max[0, 2] = 1
    # x_bounds[3]["q"][:, 2] = pose_landing_end
    # x_bounds[3]["qdot"][:, 2] = [0] * n_qdot


    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_at_first_node, pose_propulsion_start]).T, interpolation=InterpolationType.LINEAR,
               phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR,
               phase=1)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR,
               phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)

    x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
               phase=3)
    x_init.add("qdot_u", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR, phase=3)

    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=4)

    x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR, phase=5)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=5)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=5)
    # x_init.add("q", sol["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("qdot", sol["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # x_init.add("q_u", sol["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("qdot_u", sol["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # x_init.add("q", sol["q"][2], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("qdot", sol["qdot"][2], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # x_init.add("q", sol["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=5)
    # x_init.add("qdot", sol["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=5)

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
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=4)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=5)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=4)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=5)
    # u_init.add("tau", sol["tau"][0][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=2)
    # u_init.add("tau", sol["tau"][1][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=3)
    # u_init.add("tau", sol["tau"][2][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=4)
    # u_init.add("tau", sol["tau"][3][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=5)

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
        # multinode_objectives=multinode_objectives,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    model_path_2contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V2.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_2contact,
                           model_path_1contact,
                           model_path,
                           model_path,
                           model_path,
                           model_path_2contact),
        phase_time=(0.2, 0.1, 0.1, 0.4, 0.1, 0.2),
        n_shooting=(20, 10, 10, 40, 10, 20),
        min_bound=0.01,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(1000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    sol = ocp.solve(solver)

# --- Show results --- #
#     sol.print_cost()
    sol.graphs(show_bounds=True)
    save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_6phases(bio_model, sol, model_path)


if __name__ == "__main__":
    main()
