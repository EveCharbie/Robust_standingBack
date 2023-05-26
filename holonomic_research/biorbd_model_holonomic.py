from typing import Callable, Any

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, solve
from bioptim import Bounds, BiMapping, BiMappingList, BiorbdModel


class BiorbdModelCustomHolonomic(BiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints,
    very experimental and not tested
    """

    def __init__(self, bio_model: str | biorbd.Model):
        super().__init__(bio_model)
        self._holonomic_constraints = []
        self._holonomic_constraints_jacobians = []
        self._holonomic_constraints_derivatives = []
        self._holonomic_constraints_double_derivatives = []
        self.stabilization = False
        self.alpha = 0.01
        self.beta = 0.01

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        """
        Add a holonomic constraint to the model

        Parameters
        ----------
        constraint: Function | Callable[[GeneralizedCoordinates], MX]
            The holonomic constraint
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX]
            The jacobian of the holonomic constraint
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX]
            The double derivative of the holonomic constraint
        """
        self._holonomic_constraints.append(constraint)
        self._holonomic_constraints_jacobians.append(constraint_jacobian)
        self._holonomic_constraints_double_derivatives.append(constraint_double_derivative)

    @property
    def nb_holonomic_constraints(self):
        """
        Get the number of holonomic constraints

        Returns
        -------
        int
            The number of holonomic constraints
        """
        return sum([c.nnz_out() for c in self._holonomic_constraints])

    def holonomic_constraints(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints])

    def holonomic_constraints_jacobian(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints_jacobians])

    def holonomic_constraints_derivative(self, q: MX, qdot: MX):
        return self.holonomic_constraints_jacobian(q) @ qdot

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX):
        return vertcat(*[c(q, qdot, qddot) for c in self._holonomic_constraints_double_derivatives])

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        """
        Compute the forward dynamics of the model

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        tau: MX
            The generalized torques
        external_forces: MX
            The external forces
        f_contacts: MX
            The contact forces

        Returns
        -------
        MX
            The generalized accelerations
        """
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)

        mass_matrix = self.model.massMatrix(q_biorbd).to_mx()
        constraint_jacobian = self.holonomic_constraints_jacobian(q)
        constraint_jacobian_transpose = constraint_jacobian.T

        # compute the matrix DAE
        mass_matrix_augmented = horzcat(mass_matrix, constraint_jacobian_transpose)
        mass_matrix_augmented = vertcat(
            mass_matrix_augmented,
            horzcat(
                constraint_jacobian,
                MX.zeros((constraint_jacobian_transpose.shape[1], constraint_jacobian_transpose.shape[1])),
            ),
        )

        # compute b vector
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, f_ext=None, f_contacts=None).to_mx()

        biais = -self.holonomic_constraints_jacobian(qdot) @ qdot
        if self.stabilization:
            biais -= self.alpha * self.holonomic_constraints(q) + self.beta * self.holonomic_constraints_derivative(
                q, qdot
            )

        tau_augmented = vertcat(tau_augmented, biais)

        # solve with casadi Ax = b

        x = solve(mass_matrix_augmented, tau_augmented, "symbolicqr")

        return x[: self.nb_qddot]

    def dae_inverse_dynamics(
        self, q, qdot, qddot, tau, lagrange_multipliers, external_forces=None, f_contacts=None
    ) -> MX:
        """
        Compute the inverse dynamics of the model
        Ax-b = 0
        """
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)

        mass_matrix = self.model.massMatrix(q_biorbd)
        constraint_jacobian = self.holonomic_constraints_jacobian(q)
        constraint_jacobian_transpose = constraint_jacobian.T

        # compute the matrix DAE
        mass_matrix_augmented = horzcat(mass_matrix, constraint_jacobian_transpose)
        mass_matrix_augmented = vertcat(
            mass_matrix_augmented, horzcat(constraint_jacobian, MX.zeros(constraint_jacobian_transpose.shape))
        )

        # compute b vector
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, f_ext=None, f_contacts=None)
        tau_augmented = vertcat(tau_augmented, self.holonomic_constraints_jacobian(qdot) @ qdot)

        # Ax-b = 0
        return mass_matrix_augmented @ vertcat(qddot_biorbd, lagrange_multipliers) - tau_augmented
