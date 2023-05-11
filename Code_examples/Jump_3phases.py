"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.
Phase 1: Propulsion
- one contact (toe)
- objectives functions: maximize velocity of CoM and minimize time of flight

Phase 2: Phase in air
- zero contact (in the air)
- objectives functions: maximize height of CoM and maximize time of flight

Phase 3: Landing
- two contact
- objectives functions: minimize velocity CoM and minimize state qdot


"""
# --- Import package --- #

import numpy as np
import pickle
import sys

sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/lim/Documents/Anais/bioptim")
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
    )
    tau_min, tau_max, tau_init = -1000, 1000, 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3], [3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 1 (First position): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=0, min_bound=0.1, max_bound=0.3)

    # Phase 2 (Jump): Maximize height CoM + Maximize time in air
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-1000, phase=1, min_bound=0.3, max_bound=1.5)

    # Phase 3 (Landing): Minimize velocity CoM + qdot=0
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=2, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=1,
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

    # Contraint position
    pose_at_first_node = [
        -0.19,
        -0.43,
        -1.01,
        0.0044735684524460015,
        2.5999996919248574,
        -2.299999479653955,
        0.6999990764981876,
    ]
    pose_landing = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0]

    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 1
    x_bounds.add(bounds=QAndQDotBounds(bio_model[0]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Phase 2
    x_bounds.add(bounds=QAndQDotBounds(bio_model[1]))

    # Phase 3
    x_bounds.add(bounds=QAndQDotBounds(bio_model[2]))
    x_bounds[2][:, 2] = pose_landing + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * (bio_model[0].nb_tau - 3),
        [tau_max] * (bio_model[0].nb_tau - 3),
    )
    u_bounds.add(
        [tau_min] * (bio_model[1].nb_tau - 3),
        [tau_max] * (bio_model[1].nb_tau - 3),
    )
    u_bounds.add(
        [tau_min] * (bio_model[2].nb_tau - 3),
        [tau_max] * (bio_model[2].nb_tau - 3),
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * (bio_model[0].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[1].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[2].nb_tau - 3))

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
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_1C_3M.bioMod",
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_0C_3M.bioMod",
            "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_2C_3M.bioMod",
        ),
        phase_time=(0.3, 1, 0.3),
        n_shooting=(30, 100, 30),
        min_bound=50,
        max_bound=np.inf,
    )
    ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)

    # --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs()


# --- Save results --- #
# del sol.ocp
# with open(.....) as f:
#    pickle.dump()
# with open(f"Results_jump_3phases_sansrecep") as file:
#    states, controls, parameters = pickle.load(file)


if __name__ == "__main__":
    main()
