"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contact (toe)
- Objective(s) function(s): maximize velocity of CoM and minimize time of flight

Phase 1: Tucked phase
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact
- Objective(s) function(s): maximize torque

Phase 2: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): two contact (toe + heel)
- Objective(s) function(s): minimize velocity CoM
"""

# --- Import package --- #
import numpy as np
import pickle
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
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
    )
    tau_min, tau_max, tau_init = -1000, 1000, 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Propulsion): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=0, min_bound=0.1, max_bound=0.5)

    # Phase 1 (Tucked phase):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-1000, phase=1, min_bound=0.3, max_bound=2)

    # Phase 2 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=2, axes=Axis.Z)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    # Constraints
    # Phase 0 (Propulsion):
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=2,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=2,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=2,
    )

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [-0.2436, -0.2580, -0.9574, 1.0434, 2.2766, -1.8341, 0.5049]
    pose_landing = [0.0, 0.14, 6.28, 3.1, 0.0, 0.0, 0.0]

    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 1:
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_at_first_node
    x_bounds[0]["q"][:, 0] = [0] * n_qdot
    x_bounds[0]["q"].min[2, 1] = -np.pi / 2
    x_bounds[0]["q"].max[2, 1] = np.pi / 2

    # Phase 2:
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, 1] = 0
    x_bounds[1]["q"].max[2, 1] = 2 * np.pi
    x_bounds[1]["q"].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[1]["q"].max[2, 2] = 2 * np.pi + 0.5

    # Phase 3
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"][:, 2] = pose_landing
    x_bounds[2]["qdot"][:, 2] = [0] * n_qdot
    x_bounds[2]["q"].min[2, :] = 2 * np.pi - 0.5
    x_bounds[2]["q"].max[2, :] = 2 * np.pi + 0.5

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", pose_at_first_node, phase=0)
    x_init.add("qdot", [0] * n_qdot, phase=0)
    x_init.add("q", pose_at_first_node, phase=1)
    x_init.add("qdot", [0] * n_qdot, phase=1)
    x_init.add("q", pose_at_first_node, phase=2)
    x_init.add("qdot", [0] * n_qdot, phase=2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        "tau", min_bound=[tau_min] * (bio_model[0].nb_tau - 3), max_bound=[tau_max] * (bio_model[0].nb_tau - 3), phase=0
    )
    u_bounds.add(
        "tau", min_bound=[tau_min] * (bio_model[1].nb_tau - 3), max_bound=[tau_max] * (bio_model[0].nb_tau - 3), phase=1
    )
    u_bounds.add(
        "tau", min_bound=[tau_min] * (bio_model[2].nb_tau - 3), max_bound=[tau_max] * (bio_model[0].nb_tau - 3), phase=2
    )

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[1].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[2].nb_tau - 3), phase=2)

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
        variable_mappings=dof_mapping,
        n_threads=3,
    )


# --- Load model --- #
def main():
    ocp = prepare_ocp(
        biorbd_model_path=(
            "../models/Model2D_2C_3M.bioMod",
            "../models/Model2D_0C_3M.bioMod",
            "../models/Model2D_2C_3M.bioMod",
        ),
        phase_time=(0.5, 2, 0.2),
        n_shooting=(50, 200, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)

    # --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs()


if __name__ == "__main__":
    main()
