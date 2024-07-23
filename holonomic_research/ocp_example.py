"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to show how one can define its own custom dynamics function if the provided ones are not
sufficient.

More specifically this example reproduces the behavior of the DynamicsFcn.TORQUE_DRIVEN using a custom dynamics
"""

import platform

from casadi import MX, SX, vertcat, Function, jacobian
from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFcn,
    DynamicsFunctions,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    InitialGuess,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    BiMapping,
    BiMappingList,
    SelectionMapping,
    Dependency,
)
from biorbd import marker_index
from biorbd_casadi import RotoTrans
import numpy as np

from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from holonomic_research.graphs import constraints_graphs


def custom_dynamic(
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    nlp: NonLinearProgram,
    my_additional_factor=1,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.constrained_forward_dynamics(q, qdot, tau)

    # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
    # as the first argument of DynamicsEvaluation or
    # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
    # which may be useful for IRK or COLLOCATION integrators
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, my_additional_factor=1):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=True)


def generate_close_loop_constraint(
    biorbd_model, marker_1: str, marker_2: str, index: slice = slice(0, 3), local_frame_index: int = None
) -> tuple[Function, Function, Function]:
    """Generate a close loop constraint between two markers"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", biorbd_model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_1))
    marker_2_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_2))

    # if local frame is provided, the markers are expressed in the same local frame
    if local_frame_index is not None:
        jcs_t = biorbd_model.homogeneous_matrices_in_global(q_sym, local_frame_index, inverse=True)
        marker_1_sym = (jcs_t.to_mx() @ vertcat(marker_1_sym, 1))[:3]
        marker_2_sym = (jcs_t.to_mx() @ vertcat(marker_2_sym, 1))[:3]

    # the constraint is the distance between the two markers, set to zero
    constraint = (marker_1_sym - marker_2_sym)[index]
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym],
        [constraint_jacobian],
        ["q"],
        ["holonomic_constraint_jacobian"],
    ).expand()

    # the double derivative of the constraint
    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym + constraint_jacobian_func(q_dot_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "holonomic_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    custom_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    custom_dynamics: bool
        If the user wants to use custom dynamics

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModelCustomHolonomic(biorbd_model_path)
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model, "marker_1", "marker_3", index=slice(1, 3), local_frame_index=0
    )
    #
    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model.stabilization = True
    gamma = 50
    bio_model.alpha = gamma ^ 2
    bio_model.beta = 2 * gamma
    # Problem parameters
    n_shooting = 100
    final_time = 1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=1, max_bound=2)

    # Dynamics
    dynamics = DynamicsList()
    if custom_dynamics:
        dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=True)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="marker_1", second_marker="marker_3"
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS_VELOCITY, node=Node.START, first_marker="marker_1", second_marker="marker_3"
    )

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])

    # Initial guess
    q_t0 = np.array([1.54, 1.54])  # change this line for automatic initial guess that satisfies the constraints
    # q_t0 = np.array([0, 0])
    # Translations between Seg0 and Seg1 at t0, calculated with cos and sin as Seg1 has no parent
    t_t0 = np.array([np.sin(q_t0[0]), -np.cos(q_t0[0])])
    all_q_t0 = np.array([q_t0[0], t_t0[0], t_t0[1], q_t0[1], 0, 0, 0, 0])

    x_init = InitialGuess(all_q_t0)
    x_bounds[:, 0] = all_q_t0

    # Define control path constraint
    tau_min, tau_max, tau_init = -1, 1, 0

    variable_bimapping = BiMappingList()

    variable_bimapping.add("tau", to_second=[0, None, None, None], to_first=[0])
    u_bounds = Bounds([tau_min], [tau_max])
    u_init = InitialGuess([tau_init])

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        use_sx=False,
        assume_phase_dynamics=True,
        variable_mappings=variable_bimapping,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/two_pendulums.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True, _max_iter=500))

    # --- Show results --- #
    sol.animate()

    constraints_graphs(ocp, sol)


if __name__ == "__main__":
    main()
