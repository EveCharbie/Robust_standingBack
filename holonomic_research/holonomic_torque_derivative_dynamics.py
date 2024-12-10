from typing import Callable

from bioptim import (
    DynamicsFcn,
    DynamicsFunctions,
    DynamicsEvaluation,
    ConfigureProblem,
    BiMapping,
    CustomPlot,
    PlotType,
)
from casadi import MX, vertcat, Function
import numpy as np


def configure_holonomic_torque_derivative_driven(ocp, nlp, numerical_data_timeseries: dict[str, np.ndarray] = None):
    """
    Tell the program which variables are states and controls.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name = "q_u"
    names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_u,
        ocp,
        nlp,
        True,
        False,
        False,
        # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
        # see _set_kinematic_phase_mapping method
        # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
    )

    name = "qdot_u"
    names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_udot,
        ocp,
        nlp,
        True,
        False,
        False,
        # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
        # see _set_kinematic_phase_mapping method
        # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
    )

    ConfigureProblem.configure_tau(ocp, nlp, as_states=True, as_controls=False)

    ConfigureProblem.configure_taudot(ocp, nlp, as_states=False, as_controls=True)

    # extra plots
    ConfigureProblem.configure_qv(ocp, nlp, nlp.model.compute_q_v)
    ConfigureProblem.configure_qdotv(ocp, nlp, nlp.model._compute_qdot_v)
    configure_lagrange_multipliers_function(ocp, nlp, nlp.model.compute_the_lagrangian_multipliers)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, holonomic_torque_derivative_driven)


def configure_lagrange_multipliers_function(ocp, nlp, dyn_func: Callable, **extra_params):
    """
    Configure the contact points

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    dyn_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
        The function to get the values of contact forces from the dynamics
    """

    time_span_sym = vertcat(nlp.time_mx, nlp.dt_mx)
    nlp.lagrange_multipliers_function = Function(
        "lagrange_multipliers_function",
        [
            time_span_sym,
            nlp.states.scaled.mx_reduced,
            nlp.controls.scaled.mx_reduced,
            nlp.parameters.scaled.mx_reduced,
            nlp.algebraic_states.scaled.mx_reduced,
            nlp.numerical_timeseries.mx,
        ],
        [
            dyn_func(
                nlp.get_var_from_states_or_controls(
                    "q_u", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
                nlp.get_var_from_states_or_controls(
                    "qdot_u", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
                nlp.get_var_from_states_or_controls(
                    "tau", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
            )
        ],
        ["t_span", "x", "u", "p", "a", "d"],
        ["lagrange_multipliers"],
    )

    all_multipliers_names = []
    for nlp_i in ocp.nlp:
        if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
            nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
            all_multipliers_names.extend(
                [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
            )

    all_multipliers_names = [f"lagrange_multiplier_{name}" for name in all_multipliers_names]
    all_multipliers_names_in_phase = [
        f"lagrange_multiplier_{nlp.model.name_dof[i]}" for i in nlp.model.dependent_joint_index
    ]

    axes_idx = BiMapping(
        to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
    )

    nlp.plot["lagrange_multipliers"] = CustomPlot(
        lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.lagrange_multipliers_function(
            np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
        ),
        plot_type=PlotType.INTEGRATED,
        axes_idx=axes_idx,
        legend=all_multipliers_names,
    )


def holonomic_torque_derivative_driven(
    time: MX.sym,
    states: MX.sym,
    controls: MX.sym,
    parameters: MX.sym,
    algebraic_states: MX.sym,
    numerical_timeseries: MX.sym,
    nlp,
    external_forces: list = None,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX.sym
        The time of the system
    states: MX.sym
        The state of the system
    controls: MX.sym
        The controls of the system
    parameters: MX.sym
        The parameters acting on the system
    algebraic_states: MX.sym
        The algebraic states of the system
    numerical_timeseries: MX.sym
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase
    external_forces: list[Any]
        The external forces

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.states["tau"], states)
    taudot = controls
    qddot_u = nlp.model.partitioned_forward_dynamics(q_u, qdot_u, tau, external_forces=external_forces)

    return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u, taudot), defects=None)
