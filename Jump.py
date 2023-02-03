"""
The aim of this code is to create a movement a simple jump in 2 phases with a 2D model.
Phase 1:
- one contact (toe)
- objectives functions: maximize velocity of CoM and minimize time of flight

Phase 2:
- zero contact (in the air)
- objectives functions: maximize height of CoM and maximize time of flight


"""
# --- Import package --- #

import numpy as np
import sys
sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/lim/Documents/Anais/bioptim")
from bioptim import (
    BiorbdModel,
    Node,
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

    bio_model = (BiorbdModel(biorbd_model_path[0]), BiorbdModel(biorbd_model_path[1]), BiorbdModel(biorbd_model_path[2]))
    tau_min, tau_max, tau_init = -500, 500, 0
    #activation_min, activation_max, activation_init = 0, 1, 0.5
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3], [3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()
    # Phase 1: Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-0.1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=1, phase=0)

    # Phase 2: Maximize height CoM + Maximize time in air
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-0.02, phase=1, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=-1, phase=1)

    # Phase 3: Minimize velocity CoM + qdot=0
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=1, phase=2, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=1, phase=2)


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
        phase=0)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=2)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=2)

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [-0.16020787243790702, -0.3999370232092718, -1.0189630285526428, 0.0044735684524460015, 2.5999996919248574, -2.299999479653955, 0.6999990764981876]
    pose_at_end_node = [0.0, 0.0, 0.0, 3.1, 0.0, 0.0, 0.0]

    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 1
    x_bounds.add(bounds=QAndQDotBounds(bio_model[0]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Phase 2
    x_bounds.add(bounds=QAndQDotBounds(bio_model[1]))

    # Phase 3
    x_bounds.add(bounds=QAndQDotBounds(bio_model[2]))
    x_bounds[2][:, 2] = pose_at_end_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * (bio_model[0].nb_tau-3),
        [tau_max] * (bio_model[0].nb_tau-3),
    )
    u_bounds.add(
        [tau_min] * (bio_model[1].nb_tau-3),
        [tau_max] * (bio_model[1].nb_tau-3),
    )
    u_bounds.add(
        [tau_min] * (bio_model[2].nb_tau-3),
        [tau_max] * (bio_model[2].nb_tau-3),
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * (bio_model[0].nb_tau-3))
    u_init.add([tau_init] * (bio_model[1].nb_tau-3))
    u_init.add([tau_init] * (bio_model[2].nb_tau-3))

    #x_scaling = VariableScalingList()
    #x_scaling.add("q", scaling=[1, 1, 1, 1, 1, 1, 1])  # declare keys in order, so that they are concatenated in the right order
    #x_scaling.add("qdot", scaling=[1, 1, 1, 1, 1, 1, 1])

    #u_scaling = VariableScalingList()
    #u_scaling.add("tau", scaling=[1, 1, 1, 1, 1, 1, 1])

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
        #x_scaling=x_scaling,
        #u_scaling=u_scaling,
        n_threads=3)

# --- Load model --- #


def main():
    ocp = prepare_ocp(
        biorbd_model_path=("/home/lim/Documents/Anais/Robust_standingBack/Model2D.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model2D_without_contact.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model2D_2contact.bioMod"),
        phase_time=(0.2, 1, 0.2),
        n_shooting=(20, 100, 20),
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
    ocp.save(sol, "Results_jump")


if __name__ == "__main__":
    main()