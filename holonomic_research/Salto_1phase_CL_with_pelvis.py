"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis and during the tucked phase

Phase 0: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque, velocity and time

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
from visualisation import visualisation_closed_loop_1phase


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
    # states_all = []
    tau = []

    if len(sol.ns) == 1:
        q = sol.states["u"]
        qdot = sol.states["udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            q.append(sol.states[i]["u"])
            qdot.append(sol.states[i]["udot"])
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

# --- Prepare ocp --- #


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):
    bio_model = BiorbdModelCustomHolonomic(biorbd_model_path)

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="udot", weight=1, phase=0)

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

    bio_model.set_dependencies(independent_joint_index=[0, 1, 2, 5, 6, 7], dependent_joint_index=[3, 4])


    # Path constraint
    pose_salto_tendu = [0.0, 0.082, 0.0, 2.05, -1.32, 0.0]
    pose_salto_groupe = [0.13, -1.21, 0.0, 2.5013, -2.0179, 0.0]
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    mapping = BiMappingList()
    mapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    mapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model.nb_q
    n_qdot = n_q
    n_independent = bio_model.nb_independent_joints

    x_bounds = BoundsList()
    x_bounds.add("u", bounds=bio_model.bounds_from_ranges("q", mapping=mapping), phase=0)
    x_bounds.add("udot", bounds=bio_model.bounds_from_ranges("qdot", mapping=mapping), phase=0)
    x_bounds[0]["u"][:, 0] = pose_salto_tendu

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("u", np.array([pose_salto_tendu, pose_salto_groupe]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR, phase=0)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model.nb_tau - 3), phase=0)

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
        variable_mappings=mapping,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=0.2,
        n_shooting=20,
        min_bound=50,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)
    sol.graphs(show_bounds=True)

# --- Show results --- #
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_1phase(bio_model, sol, model_path)


if __name__ == "__main__":
    main()



