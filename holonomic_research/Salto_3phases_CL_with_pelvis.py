"""
...
Phase 0: Waiting phase
- zero contact
- objectives functions: minimize torque, time

Phase 1: Salto
- zero contact, holonomics constraints
- objectives functions: minimize torque, time


"""
# --- Import package --- #

import numpy as np
import pickle
# import matplotlib.pyplot as plt
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
version = 5
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

    # Phase 0 (Waiting phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=0.01, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

    # Phase 1 (Salto close loop):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="udot", weight=0.01, phase=1)

    # Phase 2 (Second flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=2, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=0.01, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)

    # Transition de phase
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0)
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

    bio_model[1].add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model[1].set_dependencies(independent_joint_index=[0, 1, 2, 5, 6], dependent_joint_index=[3, 4])

    # Path constraint
    # pose_salto_tendu_CL = [0, 0, 0, 2.2199, -1.3461]
    # pose_salto_groupe_CL = [0, 0, 0, 2.3432, -2.0252]
    # pose_salto_tendu = [0, 0, 0, 0.79, 0.70, 2.0144, -1.1036]
    # pose_salto_groupe = [0, 0, 0, 0.7966, 0.9233, 2.2199, -1.3461]
    # pose_envol_start = [0, 0, 0, 0.263, 1.6248, 2.3432, -2.0252]
    # pose_envol_final = [0, 0, 0, -0.2763, 1.0592, 1.9805, -2.0021]

    pose_takeout_end = [-0.2803, 0.4015, 0.5049, 3.0558, 1.7953, 0.2255, -0.3913]
    pose_landing_start = [-0.9554, 0.1588, 5.8322, -0.4561, 0.03, 0.6704, -0.5305]

    # pose_takeout_end = [-0.3664, 0.4059, 0.4769, 3.0406, 1.4810, 1.9299, -1.1599]
    pose_salto_start = [-0.3269, 0.6814, 0.9003, 0.35, 1.43, 2.3562, -2.3000]
    pose_salto_start_CL = [-0.3269, 0.6814, 0.9003, 2.3562, -2.3000]
    pose_salto_end = [-0.86489, 1.3925, 3.7855, 0.35, 1.14, 2.3562, -2.3000]
    pose_salto_end_CL = [-0.86489, 1.3925, 3.7855, 2.3562, -2.3000]
    # pose_landing_start = [-0.4793, 0.2286, 6.4511, -0.65854, 0.6157, 2.22312, -1.3919]

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    mapping = BiMappingList()
    dof_mapping = BiMappingList()
    mapping.add("q", to_second=[0, 1, 2, None, None, 3, 4], to_first=[0, 1, 2, 5, 6])
    mapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4], to_first=[0, 1, 2, 5, 6])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[1].nb_independent_joints

    # Phase 0: Waiting phase
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    # x_bounds.add("tau", bounds=bio_model[0].bounds_from_ranges("tau"), phase=0)
    # x_bounds[0]["q"][:, 0] = pose_salto_tendu
    x_bounds[0]["q"].min[:, 0] = np.array(pose_takeout_end) - 0.5
    x_bounds[0]["q"].max[:, 0] = np.array(pose_takeout_end)
    x_bounds[0]["q"].min[0, :] = -2
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["q"].min[1, 1:] = 0
    x_bounds[0]["q"].max[1, 1:] = 2.5
    x_bounds[0]["q"].min[2, 1] = -np.pi / 2
    x_bounds[0]["q"].max[2, 1] = 2 * np.pi
    x_bounds[0]["q"].min[2, -1] = -np.pi / 2
    x_bounds[0]["q"].max[2, -1] = 2 * np.pi + 0.5

    x_bounds[0]["qdot"].min[0, :] = -2
    x_bounds[0]["qdot"].max[0, :] = 10
    x_bounds[0]["qdot"].min[1, :] = -5
    x_bounds[0]["qdot"].max[1, :] = 10

    # Phase 1: Waiting phase
    x_bounds.add("u", bounds=bio_model[1].bounds_from_ranges("q", mapping=mapping), phase=1)
    x_bounds.add("udot", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=mapping), phase=1)
    x_bounds[1]["u"].min[0, :] = -2
    x_bounds[1]["u"].max[0, :] = 1
    x_bounds[1]["u"].min[1, 1:] = 0
    x_bounds[1]["u"].max[1, 1:] = 2.5
    x_bounds[1]["u"].min[2, 0] = -np.pi / 2
    x_bounds[1]["u"].max[2, 0] = 2 * np.pi + 0.5
    x_bounds[1]["u"].min[2, 1] = -np.pi / 2
    x_bounds[1]["u"].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[1]["u"].min[2, 2] = -np.pi / 2
    x_bounds[1]["u"].max[2, 2] = 2 * np.pi + 0.5

    x_bounds[1]["udot"].min[0, :] = -3
    x_bounds[1]["udot"].max[0, :] = 10
    x_bounds[1]["udot"].min[1, :] = -2
    x_bounds[1]["udot"].max[1, :] = 10


    # Phase 2: Second flight
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -2
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 1:] = 0
    x_bounds[2]["q"].max[1, 1:] = 2.5
    x_bounds[2]["q"].min[2, 0] = -np.pi / 2
    x_bounds[2]["q"].max[2, 0] = 2 * np.pi + 0.5
    x_bounds[2]["q"].min[2, 1] = -np.pi / 2
    x_bounds[2]["q"].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[2]["q"].min[2, -1] = 2 * np.pi - 0.5
    x_bounds[2]["q"].max[2, -1] = 2 * np.pi + 0.5
    # x_bounds[2]["q"][:, -1] = pose_landing_start

    x_bounds[2]["qdot"].min[0, :] = -2
    x_bounds[2]["qdot"].max[0, :] = 10
    x_bounds[2]["qdot"].min[1, :] = -2
    x_bounds[2]["qdot"].max[1, :] = 10


    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_takeout_end, pose_salto_start]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)

    x_init.add("u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
               phase=1)
    x_init.add("udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR, phase=1)

    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6]], phase=2)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)

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
        phase_time=(0.2, 0.3, 0.2),
        n_shooting=(20, 30, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(1)

    sol = ocp.solve(solver)
    bio_model[1].compute_external_force_holonomics_constraints(sol.states[1]["u"], sol.states[1]["udot"], sol.controls[1]["tau"])
    sol.graphs(show_bounds=True)

# --- Show results --- #
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")

    # sol.graphs(show_bounds=True)
    visualisation_closed_loop(bio_model, sol, model_path)

    # --- Compute results --- #
    # constraints_graphs(ocp, sol)
    # for index_dof in range(sol.ocp.nlp[0].model.nb_q):
    #     time = np.concatenate((sol.time[0], sol.time[1]))
    #     if index_dof < sol.ocp.nlp[1].model.nb_independent_joints:   # Graph bras
    #         fig, axes = plt.subplots(nrows=3, ncols=1)
    #         fig.suptitle(str(sol.ocp.nlp[0].model.name_dof[index_dof]))
    #         q_add = np.empty(shape=(int(time.shape[0] - sol.states[0]["q"][index_dof].shape[0])),
    #                          dtype=float)
    #         q_add[:] = np.nan
    #         axes[0].plot(time, np.concatenate((sol.states[0]["q"][index_dof], q_add), axis=0))
    #         axes[0].set_title("Q")
    #
    #         qdot_add = np.empty(shape=(int(time.shape[0] - sol.states[0]["qdot"][index_dof].shape[0])),
    #                             dtype=float)
    #         qdot_add[:] = np.nan
    #         axes[1].plot(time, np.concatenate((sol.states[0]["qdot"][index_dof], qdot_add), axis=0))
    #         axes[1].set_title("Qdot")
    #
    #         # axes[2].plot(sol.controls[0]["tau"][index_dof], time)
    #         axes[2].plot(time, (np.concatenate((sol.controls[0]["tau"][index_dof],
    #                                       sol.controls[1]["tau"][index_dof]),
    #                                      axis=0)))
    #         axes[2].set_title("Tau")
    #         axes[2].set_xlabel("Time (s)")
    #
    #         fig.tight_layout()
    #         plt.savefig("Figures/" + str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) +
    #                     "phases_V" + str(version) + str(sol.ocp.nlp[0].model.name_dof[index_dof]) + ".svg")
    #         plt.show()
    #         fig.clf()
    #
    #     else:
    #         fig, axes = plt.subplots(nrows=3, ncols=1)
    #         fig.suptitle(str(sol.ocp.nlp[0].model.name_dof[index_dof]))
    #         axes[0].plot(time, (np.concatenate((sol.states[0]["q"][index_dof],
    #                                      sol.states[1]["u"][index_dof - sol.ocp.nlp[1].model.nb_independent_joints]),
    #                                     axis=0)))
    #         axes[0].set_title("Q")
    #
    #         axes[1].plot(time, (np.concatenate((sol.states[0]["qdot"][index_dof],
    #                                      sol.states[1]["udot"][index_dof - sol.ocp.nlp[1].model.nb_independent_joints]),
    #                                     axis=0)))
    #         axes[1].set_title("Qdot")
    #
    #         axes[2].plot(time, (np.concatenate((sol.controls[0]["tau"][index_dof],
    #                                      sol.controls[1]["tau"][index_dof]),
    #                                     axis=0)))
    #         axes[2].set_title("Tau")
    #         axes[2].set_xlabel("Time (s)")
    #
    #         fig.tight_layout()
    #         plt.savefig("Figures/" + str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) +
    #                                 "phases_V" + str(version) + str(sol.ocp.nlp[0].model.name_dof[index_dof]) + ".svg")
    #         plt.show()
    #         fig.clf()


if __name__ == "__main__":
    main()



