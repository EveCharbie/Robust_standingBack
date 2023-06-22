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
#sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/mickael/Documents/Anais/bioptim")
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
    PhaseTransitionFcn,
    PhaseTransitionList,
    PhaseTransition,
)

nb_phase = 6

# --- Prepare ocp --- #


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path[0]), BiorbdModel(biorbd_model_path[1]), BiorbdModel(biorbd_model_path[2]), BiorbdModel(biorbd_model_path[3]), BiorbdModel(biorbd_model_path[4]), BiorbdModel(biorbd_model_path[5]))
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

    # Phase 1 (Preparation propulsion): Minimize tau and qdot, minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=0, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=0, derivative=True)

    # Phase 2 (Propulsion): Maximize velocity CoM + Minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-10000, phase=1, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=1, min_bound=0.01, max_bound=0.15)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=1, derivative=True)
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=10, phase=0) # Pas d'info

    # Phase 3 (Take-off): Max time and height CoM
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=2, min_bound=0.1, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END,  weight=-10000, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=2, derivative=True)

    # Phase 4 (Salto):  Rotation, Maximize
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=3, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=3, min_bound=0.3, max_bound=1.5)
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, node=Node.ALL, weight=0.01, phase=2, reference_jcs=0, derivative=True) #
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 5 (Take-off after salto): Minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-100000, phase=4, min_bound=0.01, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=4, derivative=True)

    #Phase 6 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=10000, phase=5, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10000, phase=5, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=5, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=5, derivative=True)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)


    # --- Constraints ---#
    # Constraints :
    # - contact[0]: Toe_Y
    # - contact[1]: Toe_Z
    # - contact[2]: Heel_Z
    constraints = ConstraintList()

    # Phase 1 (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=0,
        phase=0)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=2,
        phase=0)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=1,
        phase=1)

    # Phase 3 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, #ALL_SHOOTING
        node=Node.ALL_SHOOTING,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=3)

    # Phase 5 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=5)

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=5)

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [0.0, 0.14, 0.0, 3.1, 0.0, 0.0, 0.0, 0.0]
    #pose_at_first_node = [-0.2436413324142528, -0.25802995626372016, -0.9574614287717431, 0.03, 0.0, 2.2766749323100783, -1.834129725760581, 0.5049155109913805] # Position of segment during first position
    #pose_impulsion = [-0.1487, -0.2424, -0.4423, -0.7, 0.0, 1.5489, -2.0328, 0.7]
    pose_landing = [0.0, 0.14, 6.28, 3.1, 0.0, 0.0, 0.0, 0.0] # Position of segment during landing

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Pase 1: Preparation propulsion
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot # impose the first position
    x_bounds[0].min[2, 1] = -np.pi/2 # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0
    x_bounds[0].max[2, 1] = np.pi/2  # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0

    # Phase 2: Propulsion
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1].min[2, 1] = -np.pi/2 # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0
    x_bounds[1].max[2, 1] = np.pi/2 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 0


    # Phase 3: Take-off phase
    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2].min[2, 1] = -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[2].max[2, 1] = 2 * np.pi # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 1
    x_bounds[2].min[5, 0] = -0.4
    x_bounds[2].max[5, 0] = 0.85
    x_bounds[2].min[6, 0] = -0.95
    x_bounds[2].max[6, 0] = 0.02


    # Phase 4: salto
    x_bounds.add(bounds=bio_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds[3].min[2, 1] = -np.pi/2 # -np.pi/2  # range min for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[3].max[2, 1] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during middle (i.e. 1) phase 2
    x_bounds[3].min[2, 2] = 2 * np.pi - 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[3].max[2, 2] = 2 * np.pi + 0.5 # range min for q state of second segment (i.e. Pelvis RotX) during end (i.e. 2) phase 2
    x_bounds[3].min[6, :] = -2.3
    x_bounds[3].max[6, :] = -np.pi/4
    x_bounds[3].min[5, :] = 0
    x_bounds[3].max[5, :] = 3 * np.pi/4

    # Phase 5: Take-off after salto
    x_bounds.add(bounds=bio_model[4].bounds_from_ranges(["q", "qdot"]))
    x_bounds[4].min[2, :] = -np.pi/2 #2 * np.pi - 0.5
    x_bounds[4].max[2, :] = 2 * np.pi + 0.5

    # Phase 6: landing
    x_bounds.add(bounds=bio_model[5].bounds_from_ranges(["q", "qdot"]))
    x_bounds[5].min[2, :] = 2 * np.pi - 1.5   # -0.5 # range min for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[5].max[2, :] = 2 * np.pi + 0.5 # range max for q state of second segment (i.e. Pelvis RotX) during all time (i.e. :) of phase 3
    x_bounds[5][:, 2] = pose_landing + [0] * n_qdot  # impose the first position
    x_bounds[5].min[0, 2] = -1
    x_bounds[5].max[0, 2] = 1

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

    for j in range(0, nb_phase):
        u_init.add([tau_init] * (bio_model[j].nb_tau-3))

    # --- Transition phase --- #
    # Transition contact:
    # - phase 0-1
    # - phase 1-2
    # - phase 5-6
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=4)

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
        n_threads=32)

# --- Load model --- #


def main():
    ocp = prepare_ocp(
        biorbd_model_path=("/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod",
                           "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_1C_5M.bioMod",
                           "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_0C_5M.bioMod",
                           "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod"),
        phase_time=(0.3, 0.3, 0.3, 1, 0.2, 0.2),
        n_shooting=(30, 30, 30, 100, 20, 20),
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
    movement = "Salto"
    version = 10

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_states.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_states.bo", "wb") as file:
        pickle.dump(sol.states, file)

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_controls.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_controls.bo", "wb") as file:
        pickle.dump(sol.controls, file)

    ocp.save(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_parameters.bo", stand_alone=True)
    with open(str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + "_parameters.bo", "wb") as file:
        pickle.dump(sol.parameters, file)

# --- Open file pickle --- #
#    name_file = "Name_file"
#    with open(name_file, "rb") as file:
#        multi_start_0 = pickle.load(file)


if __name__ == "__main__":
    main()