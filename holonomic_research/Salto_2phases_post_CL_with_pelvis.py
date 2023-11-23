"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis and during the tucked phase (holonomic constraints)
and the preparation of landing (no holonomic constraints). And see how well the transition
between phases with and without holonomic constraints works.

Phase 0: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 1: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time
"""
# --- Import package --- #

import numpy as np
import pickle
import matplotlib.pyplot as plt
from bioptim import (
    BiorbdModel,
    Node,
    PhaseTransitionFcn,
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
    PenaltyController,
)
from casadi import MX, vertcat
from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_2phases_post


# --- Parameters --- #
movement = "Salto_close_loop_post"
version = 1
nb_phase = 2
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
            if i == 0:
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


def custom_phase_transition(
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
    bio_model = (BiorbdModelCustomHolonomic(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]))

    # Made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model[0],
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )

    bio_model[0].add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model[0].set_dependencies(independent_joint_index=[0, 1, 2, 5, 6, 7], dependent_joint_index=[3, 4])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)

    # # Phase 1 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)


    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition, phase_pre_idx=0)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Path constraint
    pose_salto_start = [-0.6369, 1.0356, 0.9974, 0.7592, 0.4129, 1.7890, -1.3444, 0.0393]
    pose_salto_start_CL = [-0.6369, 1.0356, 0.46, 1.7890, -1.3444, 0.0393]
    pose_salto_end_CL = [0.1987, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [0.1987, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]


    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    mapping = BiMappingList()
    dof_mapping = BiMappingList()
    mapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    mapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[1].nb_q
    n_qdot = n_q
    n_independent = bio_model[0].nb_independent_joints

    # Phase 0: Salto
    x_bounds = BoundsList()
    x_bounds.add("u", bounds=bio_model[0].bounds_from_ranges("q", mapping=mapping), phase=0)
    x_bounds.add("udot", bounds=bio_model[0].bounds_from_ranges("qdot", mapping=mapping), phase=0)
    x_bounds[0]["u"][:, 0] = pose_salto_start_CL
    x_bounds[0]["u"].min[0, :] = -2
    x_bounds[0]["u"].max[0, :] = 1
    x_bounds[0]["u"].min[1, 1:] = 0
    x_bounds[0]["u"].max[1, 1:] = 2.5
    x_bounds[0]["u"].min[2, 0] = 0
    x_bounds[0]["u"].max[2, 0] = np.pi / 4
    x_bounds[0]["u"].min[2, 1] = np.pi / 8
    x_bounds[0]["u"].max[2, 1] = np.pi
    x_bounds[0]["u"].min[2, 2] = np.pi / 2
    x_bounds[0]["u"].max[2, 2] = np.pi
    x_bounds[0]["udot"].min[0, :] = -3
    x_bounds[0]["udot"].max[0, :] = 10
    x_bounds[0]["udot"].min[1, :] = -2
    x_bounds[0]["udot"].max[1, :] = 10
    x_bounds[0]["udot"].min[2, :] = -5
    x_bounds[0]["udot"].max[2, :] = 10

    # Phase 1: Second flight
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[0, :] = -2
    x_bounds[1]["q"].max[0, :] = 1
    x_bounds[1]["q"].min[1, 1:] = 0
    x_bounds[1]["q"].max[1, 1:] = 2.5
    x_bounds[1]["q"].min[2, 0] = np.pi / 2
    x_bounds[1]["q"].max[2, 0] = np.pi
    x_bounds[1]["q"].min[2, 1] = np.pi / 2
    x_bounds[1]["q"].max[2, 1] = 2 * np.pi
    x_bounds[1]["q"].min[2, -1] = 2 * np.pi - 1
    x_bounds[1]["q"].max[2, -1] = 2 * np.pi + 1
    x_bounds[1]["qdot"].min[0, :] = -2
    x_bounds[1]["qdot"].max[0, :] = 10
    x_bounds[1]["qdot"].min[1, :] = -2
    x_bounds[1]["qdot"].max[1, :] = 10

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=1)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)

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
                           model_path),
        phase_time=(0.2, 0.3),
        n_shooting=(20, 30),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(100000)
    sol = ocp.solve(solver)
    sol.print_cost()

# --- Show results --- #
    sol.graphs(show_bounds=True)
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_2phases_post(bio_model, sol, model_path)

if __name__ == "__main__":
    main()



