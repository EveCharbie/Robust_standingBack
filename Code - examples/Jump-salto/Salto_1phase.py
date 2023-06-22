"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.
Phase 1: Propulsion
- one contact (toe)
- objectives functions: maximize velocity of CoM and minimize time of flight

Phase 2: Take-off phase
- zero contact
- objectives functions: maximize heigh CoM, max time

Phase 3: Salto
- zero contact
- objectives functions: maximize torque

Phase 4: Take-off after salto

Phase 5: Landing
- two contact (toe + heel)
- objectives functions: minimize velocity CoM

"""
# --- Import package --- #

import numpy as np
import pickle
import sys

sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/lim/Documents/Anais/bioptim")
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    Axis,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)

import numpy as np
from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from holonomic_research.graphs import constraints_graphs
from visualisation import visualisation_closed_loop


# --- Parameters --- #
movement = "Salto_close_loop"
version = 1
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

    if with_pelvis:

        bio_model = (BiorbdModelCustomHolonomic(biorbd_model_path))

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

        bio_model.set_dependencies(independent_joint_index=[0, 1, 2, 5, 6], dependent_joint_index=[3, 4])

        # --- Objectives functions ---#
        # Add objective functions
        objective_functions = ObjectiveList()

        # Phase 0 (Salto close loop):
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.01, max_bound=2)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

        # --- Dynamics ---#
        # Dynamics
        dynamics = DynamicsList()
        dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

        # --- Constraints ---#
        # Constraints
        constraints = ConstraintList()

        # Path constraint
        pose_salto_tendu = [0.0, 0.082, 0.0, 1.93, -1.16]
        pose_salto_groupe = [0.0, 0.082, 0.0, 2.5013, -2.0179]
        mapping = BiMappingList()
        mapping.add("tau", [None, None, None, 0, 1, 2, 3], [3, 4, 5, 6])
        tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286]
        tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286]
        tau_min = [i * 0.9 for i in tau_min_total]
        tau_max = [i * 0.9 for i in tau_max_total]
        tau_init = 0
        mapping.add("q", [0, 1, 2, None, None, 3, 4], [0, 1, 2, 5, 6])
        mapping.add("qdot", [0, 1, 2, None, None, 3, 4], [0, 1, 2, 5, 6])

        # --- Bounds ---#
        # Initialize x_bounds
        n_q = bio_model.nb_q
        n_qdot = n_q
        n_independent = bio_model.nb_independent_joints
        x_bounds = BoundsList()
        x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"], mapping=mapping))
        x_bounds[0][:, 0] = pose_salto_tendu + [0] * n_independent
        x_bounds[0][:, -1] = pose_salto_groupe + [0] * n_independent

        # Initial guess
        x_init = InitialGuessList()
        x_init.add(np.array([pose_salto_tendu + [0] * n_independent, pose_salto_groupe + [0] * n_independent]).T,
                   interpolation=InterpolationType.LINEAR)

        # Define control path constraint
        u_bounds = BoundsList()
        u_bounds.add([tau_min[3], tau_min[4], tau_min[5], tau_min[6]],
                     [tau_max[3], tau_max[4], tau_max[5], tau_max[6]])

        u_init = InitialGuessList()
        u_init.add([tau_init] * (bio_model.nb_tau - 3))

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
            # use_sx=True,
        ), bio_model

    else:
        bio_model = (BiorbdModelCustomHolonomic(biorbd_model_path))

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

        # --- Objectives functions ---#
        # Add objective functions
        objective_functions = ObjectiveList()

        # Phase 0 (Salto close loop):
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.01, max_bound=2)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)

        # --- Dynamics ---#
        # Dynamics
        dynamics = DynamicsList()
        dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

        # --- Constraints ---#
        # Constraints
        constraints = ConstraintList()

        # Path constraint
        pose_salto_tendu = [1.93, -1.16]
        pose_salto_groupe = [2.5013, -2.0179]
        mapping = BiMappingList()
        tau_min_total = [-325.531, -138, -981.1876, -735.3286]
        tau_max_total = [325.531, 138, 981.1876, 735.3286]
        tau_min = [i * 0.9 for i in tau_min_total]
        tau_max = [i * 0.9 for i in tau_max_total]
        tau_init = 0
        mapping.add("q", [None, None, 0, 1], [2, 3])
        mapping.add("qdot", [None, None, 0, 1], [2, 3])

        # --- Bounds ---#
        # Initialize x_bounds
        n_q = bio_model.nb_q
        n_qdot = n_q
        n_independent = bio_model.nb_independent_joints
        x_bounds = BoundsList()
        x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"], mapping=mapping))
        x_bounds[0][:, 0] = pose_salto_tendu + [0] * n_independent
        x_bounds[0][:, -1] = pose_salto_groupe + [0] * n_independent
        x_bounds[0][0, :].min = -np.inf
        x_bounds[0][0, :].max = np.inf
        x_bounds[0][1, :].min = -np.inf
        x_bounds[0][1, :].max = np.inf
        x_bounds[0][2, :].min = -np.inf
        x_bounds[0][2, :].max = np.inf

        # Initial guess
        x_init = InitialGuessList()
        x_init.add(np.array([pose_salto_tendu + [0] * n_independent, pose_salto_groupe + [0] * n_independent]).T,
                   interpolation=InterpolationType.LINEAR)

        # Define control path constraint
        u_bounds = BoundsList()
        u_bounds.add([tau_min[0], tau_min[1], tau_min[2], tau_min[3]],
                     [tau_max[0], tau_max[1], tau_max[2], tau_max[3]])

        u_init = InitialGuessList()
        u_init.add([tau_init] * bio_model.nb_tau)

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
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=0.3,
        n_shooting=30,
        min_bound=50,
        max_bound=np.inf,
        with_pelvis=True,
    )
    # ocp = prepare_ocp(
    #     biorbd_model_path=(str(name_folder_model) + "/" + "Model2D_4Dof_0C_5M_CL_V2.bioMod"),
    #     phase_time=0.5,
    #     n_shooting=50,
    #     min_bound=50,
    #     max_bound=np.inf,
    # )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    solver.set_tol(10e-6)
    solver.set_constraint_tolerance(1e-15)
    sol = ocp.solve(solver)

# --- Show results --- #

    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    sol.print_cost()
    # sol.graphs(show_bounds=True)
    visualisation_closed_loop(bio_model, sol, model_path)
    # --- Compute results --- #
    # constraints_graphs(ocp, sol)


if __name__ == "__main__":
    main()



