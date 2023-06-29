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
from Save import save_results_with_pickle
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
    BinodeConstraintList,
    BinodeConstraintFcn,
    BinodeConstraint,
)


# --- Parameters --- #
movement = "Salto_dedoublement"
version = 20
nb_phase_ocp1 = 7
nb_phase_ocp2 = 6
nb_phase_total = nb_phase_ocp1 + nb_phase_ocp2 - 2
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack-main/Model"


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
        BiorbdModel(biorbd_model_path[5]),
        BiorbdModel(biorbd_model_path[6]),
        BiorbdModel(biorbd_model_path[7]),
        BiorbdModel(biorbd_model_path[8]),
        BiorbdModel(biorbd_model_path[9]),
        BiorbdModel(biorbd_model_path[10]),
    )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]  # with elbow
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]  # with elbow
    tau_min = [i * 0.8 for i in tau_min_total]
    tau_max = [i * 0.8 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # First loop: phase 0:6
    # Second loop: phase 0:1-7:10

    # Phase 0 (Preparation propulsion): Minimize tau and qdot, minimize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.01, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=0, derivative=True)

    # Phase 1 (Propulsion): Maximize velocity CoM + Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=1, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=1, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=1, derivative=True)

    # Phase 2 (Wait phase, take-off): Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, phase=2, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, derivative=True, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=2, derivative=True)

    # Phase 3 (Take-off): Max time and height CoM + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=3, min_bound=0.01, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=3, derivative=True)

    # Phase 4 (Salto):  Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=4, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=4, min_bound=0.2, max_bound=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.ALL_SHOOTING, weight=-10000, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=4, derivative=True)

    # Phase 5 (Take-off after salto): Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=5, min_bound=0.001, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=5, derivative=True)

    # Phase 6 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=6, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=6, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=6, derivative=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=10000000, phase=6, axes=Axis.Y
    )

    # Phase 7 (Take-off): Maximize time and height CoM + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10000, phase=7, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=7)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=7, derivative=True)

    # Phase 8 (Salto):  Minimize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=8, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=8, min_bound=0.2, max_bound=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.ALL_SHOOTING, weight=-10000, phase=8)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=8, derivative=True)

    # Phase 9 (Take-off after salto): Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, phase=9, min_bound=0.001, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=9)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=9, derivative=True)

    # Phase 10 (Landing): Minimize CoM velocity at the end of the phase + Maximize time + Minimize tau and qdot
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=10, axes=Axis.Z)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, phase=10, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=10)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=10, derivative=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=10000000, phase=10, axes=Axis.Y
    )

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
    #     - contact[0]: Toe_Y
    #     - contact[1]: Toe_Z
    #     - contact[2]: Heel_Z
    constraints = ConstraintList()

    # Phase 0 (constraint one contact with contact 2 (i.e. toe) at the beginning of the phase 0)

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.START,
        contact_index=2,
        phase=0,
    )

    # Phase 1
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=1,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=1,
    )

    # Phase 4 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.MID,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=4,
    )

    # Phase 6 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    # NON_SLIPPING
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=6,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=6,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=6,
    )

    # Phase 8 (constraint contact between two markers during phase 3)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.MID,
        first_marker="BELOW_KNEE",
        second_marker="CENTER_HAND",
        phase=8,
    )

    # Phase 10 (constraint contact with contact 2 (i.e. toe) and 1 (i.e heel) at the end of the phase 5)
    # NON_SLIPPING
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=10,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=10,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=10,
    )

    # Transition phase
    phase_transitions = PhaseTransitionList()

    # Impact
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=5)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=9)

    # DISCONTINOUS
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=3)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=4)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=5)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=6)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=7)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=8)
    phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=9)

    # Continuity
    binode_constraints = BinodeConstraintList()

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=0,
        phase_second_idx=1,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=1,
        phase_second_idx=2,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=2,
        phase_second_idx=3,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=3,
        phase_second_idx=4,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=4,
        phase_second_idx=5,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=5,
        phase_second_idx=6,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=1,
        phase_second_idx=7,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=7,
        phase_second_idx=8,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=8,
        phase_second_idx=9,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY,
        phase_first_idx=9,
        phase_second_idx=10,
        first_node=Node.END,
        second_node=Node.START,
        key="all",
    )

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [
        0.0188,
        0.1368,
        -0.1091,
        1.78,
        0.5437,
        0.191,
        -0.1452,
        0.1821,
    ]  # Position of segment during first position
    pose_propulsion_start = [
        -0.2347217373715483,
        -0.45549996131551357,
        -0.8645258574574489,
        0.4820766547674885,
        0.03,
        2.590467089448695,
        -2.289747592408045,
        0.5538056491954265,
    ]
    pose_takeout_start = [
        -0.2777672842694191,
        0.03995514292843797,
        0.1930477703559439,
        2.589642304908377,
        0.03,
        0.5353536016159908,
        -0.8367077461678971,
        0.11196901833050495,
    ]
    pose_takeout_end = [-0.2803, 0.4015, 0.5049, 3.0558, 1.7953, 0.2255, -0.3913, -0.575]
    pose_salto_start = [
        -0.3269534844623969,
        0.681422172573302,
        0.9003344030624946,
        0.35,
        1.43,
        2.3561945135532367,
        -2.300000008273391,
        0.6999999941919349,
    ]
    pose_salto_end = [
        -0.8648803377623905,
        1.3925287774995057,
        3.785530485157555,
        0.35,
        1.14,
        2.3561945105754827,
        -2.300000018314619,
        0.6999999322366998,
    ]
    pose_landing_start = [
        -0.9554004763233065,
        0.15886445602166693,
        5.832254254152056,
        -0.45610833795726297,
        0.03,
        0.6704528346396729,
        -0.5304889643328282,
        0.654641794221728,
    ]
    pose_landing_end = [-0.9461201943294933, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]

    # --- Bounds ---#
    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Preparation propulsion
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot  # impose the first position
    x_bounds[0].min[0, 2] = -1
    x_bounds[0].max[0, 2] = 1

    # Phase 1: Propulsion
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1].min[2, 1] = -np.pi / 2
    x_bounds[1].max[2, 1] = np.pi / 2
    x_bounds[1].min[0, 2] = -1
    x_bounds[1].max[0, 2] = 1
    x_bounds[1].min[6, 2] = -np.pi / 8
    x_bounds[1].max[6, 2] = 0
    x_bounds[1].min[5, 2] = -np.pi / 8
    x_bounds[1].max[5, 2] = 0

    # Phase 2: Take-off phase (Waiting phase)
    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2].min[2, 1] = -np.pi / 2
    x_bounds[2].max[2, 1] = 2 * np.pi
    x_bounds[2].min[0, 2] = -1
    x_bounds[2].max[0, 2] = 1
    x_bounds[2].min[6, :] = -np.pi / 8
    x_bounds[2].max[6, :] = 0
    x_bounds[2].min[5, :] = -np.pi / 8
    x_bounds[2].max[5, :] = 0

    # Phase 3: Take-off phase
    x_bounds.add(bounds=bio_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds[3].min[2, 1] = -np.pi / 2
    x_bounds[3].max[2, 1] = 2 * np.pi
    x_bounds[3].min[0, 2] = -1
    x_bounds[3].max[0, 2] = 1

    # Phase 4: salto
    x_bounds.add(bounds=bio_model[4].bounds_from_ranges(["q", "qdot"]))
    x_bounds[4].min[2, 1] = -np.pi / 2
    x_bounds[4].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[4].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[4].max[2, 2] = 2 * np.pi + 0.5
    x_bounds[4].min[6, :] = -2.3
    x_bounds[4].max[6, :] = -np.pi / 4
    x_bounds[4].min[5, :] = 0
    x_bounds[4].max[5, :] = 3 * np.pi / 4
    x_bounds[4].min[0, 2] = -1
    x_bounds[4].max[0, 2] = 1

    # Phase 5: Take-off after salto
    x_bounds.add(bounds=bio_model[5].bounds_from_ranges(["q", "qdot"]))
    x_bounds[5].min[2, :] = -np.pi / 2
    x_bounds[5].max[2, :] = 2 * np.pi + 0.5
    x_bounds[5].min[0, 2] = -1
    x_bounds[5].max[0, 2] = 1

    # Phase 6: landing
    x_bounds.add(bounds=bio_model[6].bounds_from_ranges(["q", "qdot"]))
    x_bounds[6].min[2, :] = 2 * np.pi - 1.5
    x_bounds[6].max[2, :] = 2 * np.pi + 0.5
    x_bounds[6][:, 2] = pose_landing_end + [0] * n_qdot
    x_bounds[6].min[0, 2] = -1
    x_bounds[6].max[0, 2] = 1

    # Phase 7: Take-off phase
    x_bounds.add(bounds=bio_model[7].bounds_from_ranges(["q", "qdot"]))
    x_bounds[7].min[2, 1] = -np.pi / 2
    x_bounds[7].max[2, 1] = 2 * np.pi
    x_bounds[7].min[0, 2] = -1
    x_bounds[7].max[0, 2] = 1

    # Phase 8: salto
    x_bounds.add(bounds=bio_model[8].bounds_from_ranges(["q", "qdot"]))
    x_bounds[8].min[2, 1] = -np.pi / 2
    x_bounds[8].max[2, 1] = 2 * np.pi + 0.5
    x_bounds[8].min[2, 2] = 2 * np.pi - 0.5
    x_bounds[8].max[2, 2] = 2 * np.pi + 0.5
    x_bounds[8].min[6, :] = -2.3
    x_bounds[8].max[6, :] = -np.pi / 4
    x_bounds[8].min[5, :] = 0
    x_bounds[8].max[5, :] = 3 * np.pi / 4
    x_bounds[8].min[0, 2] = -1
    x_bounds[8].max[0, 2] = 1

    # Phase 9: Take-off after salto
    x_bounds.add(bounds=bio_model[9].bounds_from_ranges(["q", "qdot"]))
    x_bounds[9].min[2, :] = -np.pi / 2
    x_bounds[9].max[2, :] = 2 * np.pi + 0.5
    x_bounds[9].min[0, 2] = -1
    x_bounds[9].max[0, 2] = 1

    # Phase 10: landing
    x_bounds.add(bounds=bio_model[10].bounds_from_ranges(["q", "qdot"]))
    x_bounds[10].min[2, :] = 2 * np.pi - 1.5
    x_bounds[10].max[2, :] = 2 * np.pi + 0.5
    x_bounds[10][:, 2] = pose_landing_end + [0] * n_qdot
    x_bounds[10].min[0, 2] = -1
    x_bounds[10].max[0, 2] = 1

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(
        np.array([pose_at_first_node + [0] * n_qdot, pose_propulsion_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 0 (prepa propulsion)
    x_init.add(
        np.array([pose_propulsion_start + [0] * n_qdot, pose_takeout_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 1 (Propulsion)
    x_init.add(np.array(pose_at_first_node + [0] * n_qdot))  # Phase 2 (waiting phase)
    x_init.add(
        np.array([pose_takeout_end + [0] * n_qdot, pose_salto_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 3 (take-off)
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_salto_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 4 (salto)
    x_init.add(
        np.array([pose_salto_end + [0] * n_qdot, pose_landing_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 5 (flight)
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_landing_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 6 (landing)
    x_init.add(
        np.array([pose_takeout_start + [0] * n_qdot, pose_salto_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 7 (take-off)
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_salto_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 8 (salto)
    x_init.add(
        np.array([pose_salto_end + [0] * n_qdot, pose_landing_start + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 9 (flight)
    x_init.add(
        np.array([pose_salto_start + [0] * n_qdot, pose_landing_end + [0] * n_qdot]).T,
        interpolation=InterpolationType.LINEAR,
    )  # Phase 10 (landing)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )
    u_bounds.add(
        [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    )

    # U_bounds phase 0 and 1
    # for i in range(nb_phase_total):
    #     u_bounds.add(
    #                 [tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
    #                 [tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
    #             )

    u_init = InitialGuessList()
    u_init.add([tau_init] * (bio_model[0].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[1].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[2].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[3].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[4].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[5].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[6].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[7].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[8].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[9].nb_tau - 3))
    u_init.add([tau_init] * (bio_model[10].nb_tau - 3))

    # for i in range(nb_phase_total):
    #     u_init.add([tau_init] * (bio_model[i].nb_tau-3))

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        binode_constraints=binode_constraints,
        phase_transitions=phase_transitions,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        n_threads=32,
    )

    # --- Load model --- #


def main():
    ocp = prepare_ocp(
        biorbd_model_path=(
            str(name_folder_model) + "/" + "Model2D_8Dof_3C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_2C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_3C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_0C_5M.bioMod",
            str(name_folder_model) + "/" + "Model2D_8Dof_3C_5M.bioMod",
        ),
        phase_time=(0.2, 0.05, 0.05, 0.05, 0.4, 0.05, 0.2, 0.1, 0.4, 0.05, 0.2),
        n_shooting=(20, 5, 5, 5, 40, 5, 20, 10, 40, 5, 20),
        min_bound=50,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)

    # --- Show/Save results --- #
    save_results_with_pickle(sol, str(movement) + "_" + str(nb_phase_total) + "phases_V" + str(version) + ".pkl")
    for i in range(0, nb_phase_total):
        print("Time phase " + str(i) + ": " + str(sol.phase_time[i + 1]))
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
