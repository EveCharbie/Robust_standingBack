"""
The aim of this code is to create a backward salto in 4 phases.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact
- Objective(s) function(s): minimize torque and time

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact
- Objective(s) function(s): minimize torque and time

Phase 2: Tucked phase
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, superimpose marker (knee-hand)
- Objective(s) function(s): minimize torque and time

Phase 3: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact
- Objective(s) function(s): minimize torque and time
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
    QAndQDotBounds,
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
        BiorbdModel(biorbd_model_path[3]),
    )
    tau_min, tau_max, tau_init = -1000, 1000, 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3], to_first=[3, 4, 5, 6])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Propulsion): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=0, min_bound=0.1, max_bound=0.3)

    # Phase 1 (Flight phase): Max time and height CoM
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-1000, phase=1, min_bound=0.2, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-10000, phase=1)

    # Phase 2 (Tucked phase):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=2, min_bound=0.5, max_bound=1.5)

    # Phase 3 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=3, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=3, min_bound=0.1, max_bound=0.3)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Phase 0: Propulsion (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=2,
        phase=0,
    )

    # Phase 2: Tucked phase (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,  # ALL_SHOOTING
        node=Node.ALL_SHOOTING,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=2,
    )

    # Phase 3: Landing (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=3,
    )

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [
        -0.2436,
        -0.2580,
        -0.9574,
        1.0434,
        2.2767,
        -1.8341,
        0.5049,
    ]  # Position of segment during first position
    pose_landing = [0.0, 0.14, 6.28, 3.1, 0.0, 0.0, 0.0]  # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Propulsion
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_at_first_node  # impose the first position
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot
    x_bounds[0]["q"].min[2, 1] = -np.pi / 2
    x_bounds[0]["q"].max[2, 1] = np.pi / 2

    # Phase 1: Flight phase
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[2, 1] = -np.pi / 2
    x_bounds[1]["q"].max[2, 1] = 2 * np.pi

    # Phase 2: Tucked phase
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[2, 1] = -np.pi / 2
    x_bounds[2]["q"].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[2]["q"].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[2]["q"].max[2, 2] = 2 * np.pi + 0.5

    # Phase 3: Landing
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].min[2, :] = 2 * np.pi - 1.5
    x_bounds[3]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[3]["q"][:, 2] = pose_landing
    x_bounds[3]["qdot"][:, 2] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", pose_at_first_node, phase=0)
    x_init.add("qdot", [0] * n_qdot, phase=0)
    x_init.add("q", pose_at_first_node, phase=1)
    x_init.add("qdot", [0] * n_qdot, phase=1)
    x_init.add("q", pose_at_first_node, phase=2)
    x_init.add("qdot", [0] * n_qdot, phase=2)
    x_init.add("q", pose_at_first_node, phase=3)
    x_init.add("qdot", [0] * n_qdot, phase=3)

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
    u_bounds.add(
        "tau", min_bound=[tau_min] * (bio_model[2].nb_tau - 3), max_bound=[tau_max] * (bio_model[0].nb_tau - 3), phase=3
    )

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[1].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[2].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[3].nb_tau - 3), phase=3)

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
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_2C_3M.bioMod",
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_0C_3M.bioMod",
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_0C_3M.bioMod",
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_2C_3M.bioMod",
        ),
        phase_time=(0.5, 0.3, 1, 0.2),
        n_shooting=(50, 30, 100, 20),
        min_bound=50,
        max_bound=np.inf,
    )

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
