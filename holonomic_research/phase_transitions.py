from bioptim import PenaltyController
from casadi import MX, vertcat
from warnings import warn


def custom_phase_transition_pre(controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition from an unholonomic to a holonomic model.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: q-, qdot- = q+, qdot+
    """

    # Take the values of q of the BioMod without holonomics constraints
    states_pre = vertcat(controllers[0].states["q"].cx, controllers[0].states["qdot"].cx)

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent : nb_independent * 2]

    # Take the q of the independent joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    states_post = vertcat(q_post, qdot_post)

    return states_pre - states_post


def custom_phase_transition_post(controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition from a holonomic to an unholonomic model.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: q-, qdot- = q+, qdot+
    """

    # Take the values of q of the BioMod without holonomics constraints
    nb_independent = controllers[0].model.nb_independent_joints
    u_pre = controllers[0].states.cx[:nb_independent]
    udot_pre = controllers[0].states.cx[nb_independent : nb_independent * 2]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    q_pre = controllers[0].model.state_from_partition(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.state_from_partition(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)
    states_post = vertcat(controllers[1].states["q"].cx, controllers[1].states["qdot"].cx)

    return states_pre - states_post


def custom_takeoff(controllers: list[PenaltyController, PenaltyController]):
    """
    A discontinuous function that simulates an inelastic impact of a new contact point

    Parameters
    ----------
    transition: PhaseTransition
        A reference to the phase transition
    controllers: list[PenaltyController, PenaltyController]
            The penalty node elements

    Returns
    -------
    The difference between the last and first node after applying the impulse equations
    """

    ocp = controllers[0].ocp

    # Aliases
    pre, post = controllers
    if pre.model.nb_rigid_contacts == 0:
        warn("The chosen model does not have any rigid contact")

    q_pre = pre.states["q"].mx
    qdot_pre = pre.states["qdot"].mx

    val = []
    cx_start = []
    cx_end = []
    for key in pre.states:
        cx_end = vertcat(cx_end, pre.states[key].mapping.to_second.map(pre.states[key].cx))
        cx_start = vertcat(cx_start, post.states[key].mapping.to_second.map(post.states[key].cx))
        post_mx = post.states[key].mx
        if key == "tau":
            continuity = 0  # skip tau continuity
        else:
            continuity = post.states[key].mapping.to_first.map(pre.states[key].mx - post_mx)

        val = vertcat(val, continuity)

    name = f"PHASE_TRANSITION_{pre.phase_idx % ocp.n_phases}_{post.phase_idx % ocp.n_phases}"
    func = pre.to_casadi_func(name, val, pre.states.mx, post.states.mx)(cx_end, cx_start)
    return func
