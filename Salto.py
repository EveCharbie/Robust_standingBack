"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.
Phase 1: Propulsion
- one contact (toe)
- objectives functions: maximize velocity of CoM and minimize time of flight

Phase 2: Phase in air + salto
- zero contact (in the air)
- objectives functions: maximize height of CoM and maximize time of flight

Phase 3: Landing
- zero contact (in the air)
- objectives functions: maximize height of CoM and maximize time of flight


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
    bio_model = (BiorbdModel(biorbd_model_path[0]), BiorbdModel(biorbd_model_path[1])) #BiorbdModel(biorbd_model_path[2])
    tau_min, tau_max, tau_init = -1000, 1000, 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3], [3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 1 (First position): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=0, min_bound=0.1, max_bound=0.3)
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)

    # Phase 2 (Salto + Landing):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, weight=-100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=-10, phase=1)
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-1000, phase=1, min_bound=0.3, max_bound=1.5)
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.END, key="qdot", weight=100, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    #dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    #dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

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
        ConstraintFcn.TRACK_MARKERS,
        node=Node.END,
        marker_index=2,
        axes=Axis.Z,
        phase=1)



    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [-0.19, -0.43, -1.01, 0.0044735684524460015, 2.5999996919248574, -2.299999479653955, 0.6999990764981876]
    pose_extension = [0.0, -0.43, 0.0, 3.1, 0.0, 0.0, 0.0]
    #pose_salto = [-0.19583755162181637, 0.5015741174899331, 0.0843622312961141, 1.774819320123863, 2.19012680837638, -1.3515591584543678, -0.5899163322480979]
    #pose_salto = [0.0, 0.5, 0.9211, 1.125, 2.2382, -2.2639, 0.0]
    #pose_landing = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0]


    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 1
    x_bounds.add(bounds=QAndQDotBounds(bio_model[0]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot
    x_bounds[0].min[2, 1] = -np.pi/2 #-np.pi
    x_bounds[0].max[2, 1] = np.pi/2 #np.pi

    # Phase 2
    x_bounds.add(bounds=QAndQDotBounds(bio_model[1]))
    x_bounds[1].min[2, 1] = 0 #0
    x_bounds[1].max[2, 1] = 2 * np.pi #2 * np.pi + 0.5
    x_bounds[1].min[2, 2] = 2 * np.pi - 0.5 #0
    x_bounds[1].max[2, 2] = 2 * np.pi + 0.5
    x_bounds[1].min[4, 2] = 0 - 0.05
    x_bounds[1].max[4, 2] = 0 + 0.05

    # Phase 3
    #x_bounds.add(bounds=QAndQDotBounds(bio_model[2]))
    #x_bounds[2][:, 2] = pose_landing + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add((np.array([pose_at_first_node + [0] * n_qdot, pose_extension + [0] * n_qdot])).T, interpolation=InterpolationType.LINEAR)
    #x_init.add((np.array([pose_extension + [0] * n_qdot, pose_salto + [0] * n_qdot])).T, interpolation=InterpolationType.LINEAR)
    #x_init.add((np.array([pose_salto + [0] * n_qdot, pose_landing + [0] * n_qdot])).T, interpolation=InterpolationType.LINEAR)
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

    u_init = InitialGuessList()
    u_init.add([tau_init] * (bio_model[0].nb_tau-3))
    u_init.add([tau_init] * (bio_model[1].nb_tau-3))

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
        n_threads=3)

# --- Load model --- #


def main():
    ocp = prepare_ocp(
        biorbd_model_path=("/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_1C_3M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_0C_3M.bioMod"),
        phase_time=(0.3, 1.5),
        n_shooting=(30, 150),
        min_bound=50,
        max_bound=np.inf,
    )

    #ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)

# --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs()

# --- Save results --- #

    #del sol.ocp
    #with open(.....) as f:
    #    pickle.dump()
    #with open(f"Results_jump_3phases_sansrecep") as file:
    #    states, controls, parameters = pickle.load(file)


if __name__ == "__main__":
    main()


