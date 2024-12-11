from bioptim import (
    ObjectiveFcn,
    Node,
    Axis,
    ObjectiveList,
    PenaltyController,
)
import casadi as cas

from actuators import actuator_function


def minimize_actuator_torques(controller: PenaltyController, actuators) -> cas.MX:
    q = controller.states["q"].cx_start

    if "tau" in controller.states:
        tau = controller.states["tau"].cx_start
    else:
        tau = controller.controls["tau"].cx_start

    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(
                actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i + 3]
            ),
            actuator_function(
                actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i + 3]
            ),
        )
        out += (tau[i] / current_max_tau) ** 2
    return cas.sum1(out)


def minimize_actuator_torques_CL(controller: PenaltyController, actuators) -> cas.MX:

    nb_independent = controller.model.nb_independent_joints
    u = controller.states.cx[:nb_independent]
    v = controller.model.compute_v_from_u_explicit_symbolic(u)
    q = controller.model.state_from_partition(u, v)

    if "tau" in controller.states:
        tau = controller.states["tau"].cx_start
    else:
        tau = controller.controls["tau"].cx_start

    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(
                actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i + 3]
            ),
            actuator_function(
                actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i + 3]
            ),
        )
        out += (tau[i] / current_max_tau) ** 2
    return cas.sum1(out)


def add_objectives(objective_functions, actuators):

    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.1, max_bound=0.4, phase=0)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.01,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES,
        node=Node.END,
        weight=5,
        contact_index=1,
        quadratic=True,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES,
        node=Node.END,
        weight=5,
        contact_index=0,
        quadratic=True,
        phase=0,
    )

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=1)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=1,
    )

    # Phase 2 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-10, min_bound=0.1, max_bound=0.4, phase=2)

    # Phase 3 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=3)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=3,
    )

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=100, phase=4)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=4)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.2, max_bound=1, phase=4)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.01,
        phase=4,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=100, axes=Axis.Y, phase=4)

    return objective_functions


def add_tau_derivative_objectives(objective_functions):
    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=1)

    # Phase 2 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=2)

    # Phase 3 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=3)

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=4)

    return objective_functions


def add_taudot_objectives(objective_functions):
    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=0)

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=1)

    # Phase 2 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=2)

    # Phase 3 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=3)

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=1, phase=4)

    return objective_functions
