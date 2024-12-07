"""
The aim of this code is to create a backward salto in 5 phases with a dynamic with actuator.

Phase 0: Propulsion
- Dynamic(s): TORQUE_ACTIVATIONS_DRIVEN with contact
- Constraint(s): 2 contact
- Objective(s) function(s): minimize torque and time

Phase 1: Flight
- Dynamic(s): TORQUE_ACTIVATIONS_DRIVEN
- Constraint(s): zero contact
- Objective(s) function(s): minimize torque and time

Phase 2: Tucked phase
- Dynamic(s): TORQUE_ACTIVATIONS_DRIVEN
- Constraint(s): zero contact, superimpose marker (knee-hand)
- Objective(s) function(s): minimize torque and time

Phase 3: Preparation landing
- Dynamic(s): TORQUE_ACTIVATIONS_DRIVEN
- Constraint(s): zero contact
- Objective(s) function(s): minimize torque and time

Phase 4: Landing
- Dynamic(s): TORQUE_ACTIVATIONS_DRIVEN with contact
- Constraint(s): 2 contact
- Objective(s) function(s): minimize torque and time

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
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)

# --- Parameters --- #
nb_phase = 5


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
        BiorbdModel(biorbd_model_path[4]),
    )
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Propulsion): Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100000, phase=0, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=0, derivative=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=10, phase=0) # Pas d'info

    # Phase 1 (Flight): Max time and height CoM
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=1, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-10000, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=1, derivative=True)

    # Phase 2 (Tucked phase):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=2, min_bound=0.3, max_bound=1.5)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, node=Node.ALL, weight=0.01, phase=2, reference_jcs=0, derivative=True) #
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=2, derivative=True)

    # Phase 3 (Preparation landing): Minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=3, min_bound=0.01, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=3, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=10000, phase=4, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=4, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=4, derivative=True)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_contact=True, with_residual_torque=True)
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_residual_torque=True)
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_residual_torque=True)
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_residual_torque=True)
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_contact=True, with_residual_torque=True)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Phase 0: Propulsion (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=0,
        phase=0,
    )

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
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=0,
    )

    # Phase 3: Tucked phase (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,  # ALL_SHOOTING
        node=Node.START,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=2,
    )

    # Phase 5: Landing (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=0,
        phase=4,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=4,
    )

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0, 0.0]
    pose_landing = [0.0, 0.14, 6.28, 3.1, 0.0, 0.0, 0.0, 0.0]  # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Propulsion
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot  # impose the first position
    x_bounds[0].min[2, 1] = (
        -np.pi / 2
    )  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0
    x_bounds[0].max[2, 1] = (
        np.pi / 2
    )  # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0

    # Phase 1: Flight phase
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1].min[2, 1] = (
        -np.pi / 2
    )  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[1].max[2, 1] = (
        2 * np.pi
    )  # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1

    # Phase 2: Tucked phase
    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2].min[2, 1] = (
        -np.pi / 2
    )  # -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[2].max[2, 1] = (
        2 * np.pi + 0.5
    )  # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[2].min[2, 2] = (
        2 * np.pi - 0.5
    )  # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[2].max[2, 2] = (
        2 * np.pi + 0.5
    )  # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[2].min[6, :] = -2.3
    x_bounds[2].max[6, :] = -np.pi / 4
    x_bounds[2].min[5, :] = 0
    x_bounds[2].max[5, :] = 3 * np.pi / 4

    # Phase 3: Preparation landing
    x_bounds.add(bounds=bio_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds[3].min[2, :] = -np.pi / 2  # 2 * np.pi - 0.5
    x_bounds[3].max[2, :] = 2 * np.pi + 0.5

    # Phase 4: Landing
    x_bounds.add(bounds=bio_model[4].bounds_from_ranges(["q", "qdot"]))
    x_bounds[4].min[2, :] = (
        2 * np.pi - 1.5
    )  # -0.5 # range min for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[4].max[2, :] = (
        2 * np.pi + 0.5
    )  # range max for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[4][:, 2] = pose_landing + [0] * n_qdot  # impose the first position
    x_bounds[4].min[0, 2] = -1
    x_bounds[4].max[0, 2] = 1

    # Initial guess
    x_init = InitialGuessList()
    for x in range(nb_phase):
        x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    for j in range(0, nb_phase):
        # u_bounds.add(
        #    [-1] * (bio_model[j].nb_tau-3) + [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        #    [1] * (bio_model[j].nb_tau-3) + [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        # )
        u_bounds.add(
            [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
            [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        )

    u_init = InitialGuessList()

    for j in range(0, nb_phase):
        # u_init.add([tau_init] * (bio_model[j].nb_tau-3) * 2)
        u_init.add([tau_init] * (bio_model[j].nb_tau - 3))

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
            "../models/Model2D_8Dof_2C_5M.bioMod",
            "../models/Model2D_8Dof_0C_5M.bioMod",
            "../models/Model2D_8Dof_0C_5M.bioMod",
            "../models/Model2D_8Dof_0C_5M.bioMod",
            "../models/Model2D_8Dof_2C_5M.bioMod",
        ),
        phase_time=(0.5, 0.3, 1, 0.2, 0.2),
        n_shooting=(50, 30, 100, 20, 20),
        min_bound=0,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)

    # --- Show results --- #
    print(f"Time to solve : {sol.real_time_to_optimize}sec")
    sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)

    # --- Save results --- #
    movement = "Salto"
    version = 14

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_states.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_states.bo", "wb") as file:
        pickle.dump(sol.states, file)

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_controls.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_controls.bo", "wb") as file:
        pickle.dump(sol.controls, file)

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_parameters.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_parameters.bo", "wb") as file:
        pickle.dump(sol.parameters, file)


if __name__ == "__main__":
    main()
