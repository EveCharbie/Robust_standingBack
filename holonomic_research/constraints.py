from bioptim import (
    ConstraintFcn,
    ConstraintList,
    Node,
    Axis,
    PenaltyController,
)
from casadi import MX, vertcat
import numpy as np
from biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic


def CoM_over_toes(controller: PenaltyController) -> MX:
    q = controller.states["q"].cx_start
    CoM_pos = controller.model.center_of_mass(q)
    CoM_pos_y = CoM_pos[1]
    marker_index = controller.model.marker_index("Foot_Toe_marker")
    marker_pos = controller.model.markers(q)[marker_index]
    marker_pos_y = marker_pos[1]
    constraint = marker_pos_y - CoM_pos_y
    return constraint


def add_constraints(constraints) -> ConstraintList:
    """Phase 0 (Propulsion) and Phase 4 (Landing) constraints"""
    # Phase 0 (Propulsion):
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        marker_index="Foot_Toe_marker",
        node=Node.START,
        phase=0,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    MAX_TAKE_OFF_FORCE_THRESHOLD = 25
    MIN_TAKE_OFF_FORCE_THRESHOLD = 0.01

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        node=Node.END,
        contact_index=0,
        phase=0,
        min_bound=MIN_TAKE_OFF_FORCE_THRESHOLD,
        max_bound=MAX_TAKE_OFF_FORCE_THRESHOLD,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        node=Node.END,
        contact_index=1,
        phase=0,
        min_bound=MIN_TAKE_OFF_FORCE_THRESHOLD,
        max_bound=MAX_TAKE_OFF_FORCE_THRESHOLD,
    )

    # Phase 4 (Landing):
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        marker_index="Foot_Toe_marker",
        node=Node.START,
        phase=4,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=4,
    )
    return constraints


def custom_contraint_lambdas_normal(controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    """The model can only pull on the legs, not push"""
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx if "tau" in controller.controls else controller.states["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    return lagrange_0


def custom_contraint_lambdas_shear(controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    """
    Relaxed friction cone, the model can push a little bit
    lagrange_1**2 < lagrange_0**2
    """
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx if "tau" in controller.controls else controller.states["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)
    lagrange_0 = lambdas[0]
    lagrange_1 = lambdas[1]

    return lagrange_0**2 - lagrange_1**2


def add_constraint_tucking_friction_cone(biomodel_holonomic, constraints):
    # "relaxed friction cone"
    constraints.add(
        custom_contraint_lambdas_shear,
        node=Node.ALL_SHOOTING,
        bio_model=biomodel_holonomic,
        max_bound=np.inf,
        min_bound=0,
        phase=2,
    )
    # The model can only pull on the legs, not push
    constraints.add(
        custom_contraint_lambdas_normal,
        node=Node.ALL_SHOOTING,
        bio_model=biomodel_holonomic,
        max_bound=-0.1,
        min_bound=-np.inf,
        phase=2,
    )
    return constraints
