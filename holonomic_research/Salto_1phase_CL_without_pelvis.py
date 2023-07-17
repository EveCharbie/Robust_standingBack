"""
The aim of this code is to ...
Phase 0: Tuck phase
- zero contact, holonomics constraints
- objectives functions: maximize torque, minimize time
"""
# --- Import package --- #

import numpy as np
import pickle
from bioptim import (
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)

from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic


# --- Parameters --- #
movement = "Salto_close_loop"
version = 2
nb_phase = 1
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
            q.append(sol.states[i]["u"])
            qdot.append(sol.states[i]["udot"])
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

# --- Prepare ocp --- #


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound, with_pelvis: False):
    bio_model = (BiorbdModelCustomHolonomic(biorbd_model_path))

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Salto close loop):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=0, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model,
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )

    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model.set_dependencies(independent_joint_index=[2, 3], dependent_joint_index=[0, 1])

    # Path constraint
    pose_salto_tendu_CL = [2.2199, -1.3461]
    pose_salto_groupe_CL = [2.3432, -2.0252]
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
    n_q = bio_model.nb_q
    n_qdot = n_q
    n_independent = bio_model.nb_independent_joints

    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q", mapping=mapping), phase=0)
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot", mapping=mapping), phase=0)
    x_bounds[0]["q"][:, 0] = pose_salto_tendu_CL
    x_bounds[0]["qdot"][:, 0] = [0] * n_independent
    x_bounds[0]["q"][:, -1] = pose_salto_groupe_CL
    x_bounds[0]["qdot"][:, -1] = [0] * n_independent

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_salto_tendu_CL, pose_salto_groupe_CL]).T,
               interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", np.array([[0] * n_independent, [0] * n_independent]).T,
               interpolation=InterpolationType.LINEAR, phase=0)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                 max_bound=[tau_max[0], tau_max[1], tau_max[2], tau_max[3]], phase=0)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model.nb_tau, phase=0)

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
        variable_mappings=mapping,
        n_threads=32,
        assume_phase_dynamics=True,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_4Dof_0C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=0.5,
        n_shooting=50,
        min_bound=50,
        max_bound=np.inf,
        with_pelvis=True,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)
    sol.graphs(show_bounds=True)

# --- Show results --- #

    # save_results(sol, str(movement) + "_" + "without_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    # sol.print_cost()
    # visualisation_closed_loop(bio_model, sol, model_path)


if __name__ == "__main__":
    main()



