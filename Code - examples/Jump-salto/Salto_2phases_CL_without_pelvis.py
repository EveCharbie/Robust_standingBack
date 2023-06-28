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
version = 4
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
        states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 1:
                q.append(sol.states[i]["u"])
                qdot.append(sol.states[i]["udot"])
                states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])
            else:
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
    states_pre = controllers[0].states.cx

    nb_q = controllers[0].model.nb_q
    q_pre = controllers[0].states.cx[:nb_q]
    qdot_pre = controllers[0].states.cx[nb_q:]

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


# --- Prepare ocp --- #

def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[1]))

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Waiting phase):
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS_VELOCITY,
    #                         node=Node.END, marker_index=2, weight=10, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
    #                         node=Node.END, first_marker="BELOW_KNEE", second_marker="CENTER_HAND", phase=0)

    # Phase 1 (Salto close loop):
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False, phase=1)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(custom_phase_transition, phase_pre_idx=0)

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

    bio_model[1].set_dependencies(independent_joint_index=[2, 3], dependent_joint_index=[0, 1])

    # Path constraint
    pose_salto_tendu_CL = [2.2199, -1.3461]
    pose_salto_groupe_CL = [2.3432, -2.0252]
    pose_salto_tendu = [0.79, 0.70, 2.0144, -1.1036]
    pose_salto_groupe = [0.7966, 0.9233, 2.2199, -1.3461]
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

    # Phase 0: Waiting phase
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_salto_tendu
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot

    # Phase 1 : Salto
    # x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"], mapping=mapping))
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q", mapping=mapping), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=mapping), phase=1)
    x_bounds[1]["q"][:, 0] = pose_salto_tendu_CL
    x_bounds[1]["qdot"][:, 0] = [0] * n_independent
    x_bounds[1]["q"][:, -1] = pose_salto_groupe_CL
    x_bounds[1]["qdot"][:, -1] = [0] * n_independent

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_salto_tendu, pose_salto_groupe]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", np.array([pose_salto_tendu_CL, pose_salto_groupe_CL]).T,
               interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("qdot", np.array([[0] * n_independent, [0] * n_independent]).T,
               interpolation=InterpolationType.LINEAR, phase=1)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=1)


    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=1)

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
                           model_path),
        phase_time=(0.2, 0.3),
        n_shooting=(20, 30),
        min_bound=50,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    # ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(1000)
    solver.set_tol(10e-6)
    solver.set_constraint_tolerance(1e-8)
    sol = ocp.solve(solver)

# --- Show results --- #
#     save_results(sol, str(movement) + "_" + "without_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    # sol.print_cost()

    # sol.graphs(show_bounds=True)
    # visualisation_closed_loop(bio_model, sol, model_path)

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



