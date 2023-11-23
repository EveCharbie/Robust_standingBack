"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis during the flight phase (no holonomic constraints),
the tucked phase (holonomic constraints) and
the preparation of landing (no holonomic constraints).
We also want to see how well the transition
between phases with and without holonomic constraints works.

Phase 0: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 2: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

"""
# --- Import package --- #
import matplotlib.pyplot as plt
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from bioptim import (
    ConstraintFcn,
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
    BoundsList,
    InitialGuessList,
    Solver,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
    HolonomicBiorbdModel,
    PenaltyController,
    PhaseTransitionFcn,
    QuadratureRule,
    PenaltyOption,
    DynamicsFunctions,
)

from casadi import MX, vertcat, Function
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_3phases


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

    if len(sol.ns) == 1:
        q = sol.states["q_u"]
        qdot = sol.states["q_udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 1:
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
def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp

# --- Parameters --- #
movement = "Salto_close_loop"
version = 19
nb_phase = 3
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"
pickle_sol_init = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_close_loop_with_pelvis_3phases_V17.pkl"

index_holonomics_constraints = 1
independent_joint_index = [0, 1, 2, 5, 6, 7]
dependent_joint_index = [3, 4]

sol = get_created_data_from_pickle(pickle_sol_init)
# q_init_holonomic = sol["q"][index_holonomics_constraints][independent_joint_index]
# qdot_init_holonomic = sol["qdot"][index_holonomics_constraints][independent_joint_index]

phase_time_init = []
for i in range(len(sol["time"])):
    time_final = sol["time"][i][-1] - sol["time"][i][0]
    phase_time_init.append(time_final)

n_shooting_init = []
for i in range(len(sol["time"])):
    n_shooting_final = sol["time"][i].shape[0] - 1
    n_shooting_init.append(n_shooting_final)


# --- Prepare ocp --- #

def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound) -> (HolonomicBiorbdModel, OptimalControlProgram):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    variable_bimapping = BiMappingList()
    dof_mapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Flight phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True,  weight=10, phase=0)

    # Phase 1 (Tucked phase:
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.01, phase=1)

    # Phase 2 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=2, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=2)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(
        bio_model[1].holonomic_torque_driven,
        dynamic_function=DynamicsFunctions.holonomic_torque_driven,
        mapping=variable_bimapping,
    )
    # dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=0)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=1)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    holonomic_constraints = HolonomicConstraintsList()

    # Phase 0: Take-off
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[1],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
    )
    # Made up constraints

    bio_model[1].set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )

    # Path constraint

    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_salto_start_CL = [-0.6369, 1.0356, 1.5062, 2.1667, -1.9179, 0.0393]
    pose_salto_end_CL = [0.1987, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [0.1987, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[1].nb_independent_joints

    # Phase 0: Flight phase
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_takeout_start
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["q"].min[1, 1:] = 0
    x_bounds[0]["q"].max[1, 1:] = 2.5
    x_bounds[0]["q"].min[2, 1] = -np.pi / 4
    x_bounds[0]["q"].max[2, 1] = np.pi / 2
    x_bounds[0]["q"].min[2, -1] = -np.pi / 2
    x_bounds[0]["q"].max[2, -1] = np.pi / 2
    x_bounds[0]["q"].min[4, -1] = 1

    x_bounds[0]["qdot"].min[0, :] = -5
    x_bounds[0]["qdot"].max[0, :] = 5
    x_bounds[0]["qdot"].min[1, :] = -2
    x_bounds[0]["qdot"].max[1, :] = 10
    x_bounds[0]["qdot"].min[2, :] = -5
    x_bounds[0]["qdot"].max[2, :] = 5

    # Phase 1: Tucked phase
    x_bounds.add("q_u", bounds=bio_model[1].bounds_from_ranges("q", mapping=variable_bimapping), phase=1)
    x_bounds.add("qdot_u", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=variable_bimapping), phase=1)
    x_bounds[1]["q_u"].min[0, :] = -2
    x_bounds[1]["q_u"].max[0, :] = 1
    x_bounds[1]["q_u"].min[1, 1:] = 0
    x_bounds[1]["q_u"].max[1, 1:] = 2.5
    x_bounds[1]["q_u"].min[2, 0] = 0
    x_bounds[1]["q_u"].max[2, 0] = np.pi / 2
    x_bounds[1]["q_u"].min[2, 1] = np.pi / 8
    x_bounds[1]["q_u"].max[2, 1] = 2 * np.pi
    x_bounds[1]["q_u"].min[2, 2] = 3/4 * np.pi
    x_bounds[1]["q_u"].max[2, 2] = 3/2 * np.pi
    x_bounds[1]["qdot_u"].min[0, :] = -5
    x_bounds[1]["qdot_u"].max[0, :] = 5
    x_bounds[1]["qdot_u"].min[1, :] = -2
    x_bounds[1]["qdot_u"].max[1, :] = 10
    x_bounds[1]["q_u"].max[3, :] = 2.6
    x_bounds[1]["q_u"].min[3, :] = 1.30

    # Phase 2: Preparation landing
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -2
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 1:] = 0
    x_bounds[2]["q"].max[1, 1:] = 2.5
    x_bounds[2]["q"].min[2, :] = 3/4 * np.pi
    x_bounds[2]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[2]["qdot"].min[0, :] = -5
    x_bounds[2]["qdot"].max[0, :] = 5
    x_bounds[2]["qdot"].min[1, :] = -10
    x_bounds[2]["qdot"].max[1, :] = 10
    x_bounds[2]["q"].max[:, -1] = np.array(pose_landing_start) + 0.5
    x_bounds[2]["q"].min[:, -1] = np.array(pose_landing_start) - 0.5

    # Initial guess
    x_init = InitialGuessList()
    # x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR, phase=0)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q_u", sol["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q_udot", sol["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)

    # x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    # x_init.add("q_udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    x_init.add("q_u", sol["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q_udot", sol["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)

    # x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=2)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", sol["q"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot", sol["qdot"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=2)

    u_init = InitialGuessList()
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", sol["tau"][0][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol["tau"][1][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", sol["tau"][2][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=2)

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
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path,
                           model_path,
                           model_path),
        # phase_time=(0.2, 0.3, 0.2),
        # n_shooting=(20, 30, 20),
        phase_time=(phase_time_init[0],
                    phase_time_init[1],
                    phase_time_init[2],
                    ),
        n_shooting=(n_shooting_init[0],
                    n_shooting_init[1],
                    n_shooting_init[2],
                    ),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol = ocp.solve(solver)
    sol.graphs()

# --- Show results --- #
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_3phases(bio_model, sol, model_path)
    sol.graphs(show_bounds=True)

    # Graph to compute forces during holonomic constraints body-body
    q, qdot, qddot, lambdas = compute_all_states(sol, bio_model[1], index_holonomics_constraints=index_holonomics_constraints)

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


if __name__ == "__main__":
    main()
