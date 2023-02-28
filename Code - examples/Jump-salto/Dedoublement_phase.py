"""
The aim of this code is to create a backward salto with two different technique:
- the first one with a wait phase to simulate an error timing before the salto
- the second technique with the optimal technique

Phase 0: Preparation propulsion
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize time, tau and qdot derivative
- Dynamics: with_contact

Phase 1: Propulsion
- 2 contacts (TOE_Y, TOE_Z)
- Objectives functions: minimize time, velocity of CoM at the end of the phase, tau and qdot derivative
- Dynamics: with_contact

Phase 2: Wait phase (Take-off phase)
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 3: Take-off phase
- 0 contact
- Objectives functions: maximize heigh CoM, time, and minimize tau and qdot derivative

Phase 4: Salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 5: Take-off after salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 6: Landing
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize velocity CoM at the end, minimize tau and qdot derivative

Phase 7: Take-off phase
- 0 contact
- Objectives functions: maximize heigh CoM, time, and minimize tau and qdot derivative

Phase 8: Salto
- 0 contact
- Objectives functions: minimize tau and qdot derivative

Phase 9: Take-off after salto
- 0 contact
- Objectives functions: maximize max time, minimize tau and qdot derivative

Phase 10: Landing
- 3 contacts (TOE_Y, TOE_Z, HEEL_Z)
- Objectives functions: minimize velocity CoM at the end, minimize tau and qdot derivative

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
    PhaseTransitionList,
    PhaseTransitionFcn,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
)

# --- Parameters --- #
movement = "Salto"
version = 1
nb_phase = 11


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

    data = dict(states=sol.states,
                controls=sol.controls,
                parameters=sol.parameters,
                iterations=sol.iterations,
                cost=sol.cost,
                detailed_cost=sol.detailed_cost,
                real_time_to_optimize=sol.real_time_to_optimize,
                status=sol.status)

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 BiorbdModel(biorbd_model_path[3]),
                 BiorbdModel(biorbd_model_path[4]),
                 BiorbdModel(biorbd_model_path[5]),
                 BiorbdModel(biorbd_model_path[6]),
                 BiorbdModel(biorbd_model_path[7]),
                 BiorbdModel(biorbd_model_path[8]),
                 BiorbdModel(biorbd_model_path[9]),
                 BiorbdModel(biorbd_model_path[10])
                 )
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7])
    #dof_mapping.add("residual_tau", [None, None, None, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7])


    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # First technique: phase 0:6
    # Second technique: phase 0:1-7:10

    # Phase 0 (Preparation propulsion): Minimize tau and qdot, minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=0, derivative=True)

    # Phase 1 (Propulsion): Maximize velocity CoM + Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=1, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100000, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=1, derivative=True)

    # Phase 2 (Wait phase, take-off): Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=2, derivative=True)

    # Phase 3 (Take-off): Max time and height CoM + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=3, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END,  weight=-10000, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=3, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 4 (Salto):  Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=4, min_bound=0.3, max_bound=1.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=4, derivative=True)

    # Phase 5 (Take-off after salto): Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=5, min_bound=0.01, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=5, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=5, derivative=True)

    #Phase 6 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=10000, phase=6, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=6, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=6, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=6, derivative=True)

    # Phase 7 (Take-off): Maximize time and height CoM + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=7, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END,  weight=-10000, phase=7)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=7, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=7, derivative=True)

    # Phase 8 (Salto):  Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=8, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=8, min_bound=0.3, max_bound=1.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=8, derivative=True)

    # Phase 9 (Take-off after salto): Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=9, min_bound=0.01, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=9, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=9, derivative=True)

    #Phase 10 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=10000, phase=10, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=10, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=10, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=10, derivative=True)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)


    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    # Phase 0 (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
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
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=0)

    # Phase 1
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=1,
        phase=1)

    # Phase 4 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=4)

    # Phase 6 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=6)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=6)

    # Phase 8 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=8)

    # Phase 10 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=10)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=10)

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0, 0.0]
    pose_landing = [0.0, 0.14, 6.28, 3.1, 0.0, 0.0, 0.0, 0.0] # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Preparation propulsion
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot # impose the first position
    x_bounds[0].min[2, 1] = -np.pi/2 # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0
    x_bounds[0].max[2, 1] = np.pi/2 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0

    # Phase 1: Propulsion
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1][:, 0] = pose_at_first_node + [0] * n_qdot # impose the first position
    x_bounds[1].min[2, 1] = -np.pi/2 # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0
    x_bounds[1].max[2, 1] = np.pi/2 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0

    # Phase 2: Take-off phase
    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2].min[2, 1] = -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[2].max[2, 1] = 2 * np.pi # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1

    # Phase 3: Take-off phase
    x_bounds.add(bounds=bio_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds[3].min[2, 1] = -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[3].max[2, 1] = 2 * np.pi # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1

    # Phase 4: salto
    x_bounds.add(bounds=bio_model[4].bounds_from_ranges(["q", "qdot"]))
    x_bounds[4].min[2, 1] = -np.pi/2 # -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[4].max[2, 1] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[4].min[2, 2] = 2 * np.pi - 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[4].max[2, 2] = 2 * np.pi + 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[4].min[6, :] = -2.3
    x_bounds[4].max[6, :] = -np.pi/4
    x_bounds[4].min[5, :] = 0
    x_bounds[4].max[5, :] = 3 * np.pi/4

    # Phase 5: Take-off after salto
    x_bounds.add(bounds=bio_model[5].bounds_from_ranges(["q", "qdot"]))
    x_bounds[5].min[2, :] = -np.pi/2 #2 * np.pi - 0.5
    x_bounds[5].max[2, :] = 2 * np.pi + 0.5

    # Phase 6: landing
    x_bounds.add(bounds=bio_model[6].bounds_from_ranges(["q", "qdot"]))
    x_bounds[6].min[2, :] = 2 * np.pi - 1.5   # -0.5 # range min for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[6].max[2, :] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[6][:, 2] = pose_landing + [0] * n_qdot  # impose the first position
    x_bounds[6].min[0, 2] = -1
    x_bounds[6].max[0, 2] = 1

    # Phase 7: Take-off phase
    x_bounds.add(bounds=bio_model[7].bounds_from_ranges(["q", "qdot"]))
    x_bounds[7].min[2, 1] = -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[7].max[2, 1] = 2 * np.pi # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1

    # Phase 8: salto
    x_bounds.add(bounds=bio_model[8].bounds_from_ranges(["q", "qdot"]))
    x_bounds[8].min[2, 1] = -np.pi/2 # -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[8].max[2, 1] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[8].min[2, 2] = 2 * np.pi - 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[8].max[2, 2] = 2 * np.pi + 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[8].min[6, :] = -2.3
    x_bounds[8].max[6, :] = -np.pi/4
    x_bounds[8].min[5, :] = 0
    x_bounds[8].max[5, :] = 3 * np.pi/4

    # Phase 9: Take-off after salto
    x_bounds.add(bounds=bio_model[9].bounds_from_ranges(["q", "qdot"]))
    x_bounds[9].min[2, :] = -np.pi/2 #2 * np.pi - 0.5
    x_bounds[9].max[2, :] = 2 * np.pi + 0.5

    # Phase 10: landing
    x_bounds.add(bounds=bio_model[10].bounds_from_ranges(["q", "qdot"]))
    x_bounds[10].min[2, :] = 2 * np.pi - 1.5   # -0.5 # range min for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[10].max[2, :] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[10][:, 2] = pose_landing + [0] * n_qdot  # impose the first position
    x_bounds[10].min[0, 2] = -1
    x_bounds[10].max[0, 2] = 1

    # Initial guess
    x_init = InitialGuessList()
    for x in range(nb_phase):
        x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    for j in range(0, nb_phase):
        u_bounds.add(
            [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
            [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        )

    u_init = InitialGuessList()

    # Transition phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=5)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1, phase_post_idx=7)

    for j in range(0, nb_phase):
        u_init.add([tau_init] * (bio_model[j].nb_tau-3))

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
        biorbd_model_path=("/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_1C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod",
                           ),
        phase_time=(0.3, 0.15, 0.3, 0.3, 1, 0.2, 0.2, 0.3, 1, 0.2, 0.2),
        n_shooting=(30, 15, 30, 30, 100, 20, 20, 30, 100, 20, 20),
        min_bound=0,
        max_bound=np.inf,
    )

    #ocp.add_plot_penalty()

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
    save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version))


if __name__ == "__main__":
    main()