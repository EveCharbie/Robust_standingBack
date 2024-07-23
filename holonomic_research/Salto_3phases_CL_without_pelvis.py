"""
The aim of this code is to test the holonomic constraint of the flight phase
without the pelvis during the flight phase (no holonomic constraints),
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

import numpy as np
import pickle
import matplotlib.pyplot as plt
from bioptim import (
    BiorbdModel,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionFcn,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    PenaltyController,
)
from casadi import MX, vertcat
from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop


# --- Parameters --- #
movement = "Salto_close_loop"
version = 3
nb_phase = 3
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"


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
        q = sol.states["u"]
        qdot = sol.states["udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 1:
                q.append(sol.states[i]["u"])
                qdot.append(sol.states[i]["udot"])
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
    q_post = controllers[1].model.q_from_u_and_v(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.q_from_u_and_v(udot_post, vdot_post)

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
    q_pre = controllers[0].model.q_from_u_and_v(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.q_from_u_and_v(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)

    states_post = controllers[1].states.cx

    return states_pre - states_post


# --- Prepare ocp --- #

def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 )

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Flight phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

    # Phase 1 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.000001, phase=1)

    # Phase 2 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=2, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=0)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=1)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model[1],
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )
    #
    bio_model[1].add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model[1].set_dependencies(independent_joint_index=[2, 3], dependent_joint_index=[0, 1])

    # Path constraint
    pose_salto_tendu_CL = [2.2199, -1.3461]
    pose_salto_groupe_CL = [2.3432, -2.0252]
    pose_salto_tendu = [0.79, 0.70, 2.0144, -1.1036]
    pose_salto_groupe = [0.7966, 0.9233, 2.2199, -1.3461]
    pose_envol_start = [0.263, 1.6248, 2.3432, -2.0252]
    pose_envol_final = [-0.2763, 1.0592, 1.9805, -2.0021]
    tau_min_total = [-325.531, -138, -981.1876, -735.3286]
    tau_max_total = [325.531, 138, 981.1876, 735.3286]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    mapping = BiMappingList()
    mapping.add("q", to_second=[None, None, 0, 1], to_first=[2, 3])
    mapping.add("qdot", to_second=[None, None, 0, 1], to_first=[2, 3])

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[1].nb_independent_joints

    # Phase 0: Flight phase
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_salto_tendu
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot

    # Phase 1 : Tucked phase
    x_bounds.add("u", bounds=bio_model[1].bounds_from_ranges("q", mapping=mapping), phase=1)
    x_bounds.add("udot", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=mapping), phase=1)

    # Phase 2: Preparation landing
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"][:, -1] = pose_envol_final
    x_bounds[2]["qdot"][:, -1] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_salto_tendu, pose_salto_groupe]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("u", np.array([pose_salto_tendu_CL, pose_salto_groupe_CL]).T,
               interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("udot", np.array([[0] * n_independent, [0] * n_independent]).T,
               interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", np.array([pose_envol_start, pose_envol_final]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=2)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=1)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=2)

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
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_4Dof_0C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path,
                           model_path,
                           model_path),
        phase_time=(0.2, 0.3, 0.2),
        n_shooting=(20, 30, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(1000)
    solver.set_tol(10e-6)
    solver.set_constraint_tolerance(1e-8)
    sol = ocp.solve(solver)

# --- Show results --- #
    sol.graphs(show_bounds=True)
    save_results(sol, str(movement) + "_" + "without_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop(bio_model, sol, model_path)


if __name__ == "__main__":
    main()



