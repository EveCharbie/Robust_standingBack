from typing import Callable, Any
import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
import casadi as cas
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from biorbd import marker_index, segment_index
from casadi import MX, DM, vertcat, horzcat, Function, solve, rootfinder, inv_minor, inv, fmod, pi, transpose
from bioptim import BiorbdModel
import numpy as np


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
        self._dependent_joint_index = []
        self._independent_joint_index = [i for i in range(self.nb_q)]

    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        """Set the dependencies between the joints of the model"""
        if len(dependent_joint_index) + len(independent_joint_index) != self.nb_q:
            raise RuntimeError(
                "The sum of the number of dependent and independent joints should be equal to the number of DoF of the model"
            )

        self._dependent_joint_index = dependent_joint_index
        self._independent_joint_index = independent_joint_index

    @property
    def nb_independent_joints(self):
        return len(self._independent_joint_index)

    @property
    def nb_dependent_joints(self):
        return len(self._dependent_joint_index)

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
        Compute the forward dynamics of the model, with full implicit formulation

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

    def partitioned_mass_matrix(self, q):
        """
        This function returns the partitioned mass matrix, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned mass matrix, reordered in function independent and dependent joints
        """
        # u: independent
        # v: dependent
        mass_matrix = self.model.massMatrix(q).to_mx()
        mass_matrix_uu = mass_matrix[self._independent_joint_index, self._independent_joint_index]
        mass_matrix_uv = mass_matrix[self._independent_joint_index, self._dependent_joint_index]
        mass_matrix_vu = mass_matrix[self._dependent_joint_index, self._independent_joint_index]
        mass_matrix_vv = mass_matrix[self._dependent_joint_index, self._dependent_joint_index]

        first_line = horzcat(mass_matrix_uu, mass_matrix_uv)
        second_line = horzcat(mass_matrix_vu, mass_matrix_vv)

        return vertcat(first_line, second_line)

    def partitioned_non_linear_effect(self, q, qdot, f_ext=None, f_contacts=None):
        """
        This function returns the partitioned non linear effect, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        """

        non_linear_effect = self.model.NonLinearEffect(q, qdot, f_ext=f_ext, f_contacts=f_contacts).to_mx()
        non_linear_effect_u = non_linear_effect[self._independent_joint_index]
        non_linear_effect_v = non_linear_effect[self._dependent_joint_index]

        return vertcat(non_linear_effect_u, non_linear_effect_v)

    def partitioned_q(self, q):
        """
        This function returns the partitioned q, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned q, reorder in function independent and dependent joints
        """
        q_u = q[self._independent_joint_index]
        q_v = q[self._dependent_joint_index]

        return vertcat(q_u, q_v)

    def partitioned_qdot(self, qdot):
        """
        This function returns the partitioned qdot, reordered in function independent and dependent joints

        Parameters
        ----------
        qdot: MX
            The generalized velocities

        Returns
        -------
        MX
            The partitioned qdot, reordered in function independent and dependent joints
        """
        qdot_u = qdot[self._independent_joint_index]
        qdot_v = qdot[self._dependent_joint_index]

        return vertcat(qdot_u, qdot_v)

    def partitioned_tau(self, tau):
        """
        This function returns the partitioned tau, reordered in function independent and dependent joints

        Parameters
        ----------
        tau: MX
            The generalized torques

        Returns
        -------
        MX
            The partitioned tau, reordered in function independent and dependent joints
        """
        tau_u = tau[self._independent_joint_index]
        tau_v = tau[self._dependent_joint_index]

        return vertcat(tau_u, tau_v)

    def partitioned_constrained_jacobian(self, q):
        """
        This function returns the partitioned constrained jacobian, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned constrained jacobian, reordered in function independent and dependent joints
        """
        constrained_jacobian = self.holonomic_constraints_jacobian(q)
        constrained_jacobian_u = constrained_jacobian[:, self._independent_joint_index]
        constrained_jacobian_v = constrained_jacobian[:, self._dependent_joint_index]

        return horzcat(constrained_jacobian_u, constrained_jacobian_v)

    def forward_dynamics_constrained_independent(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        """
        This is the forward dynamics of the model, but only for the independent joints

        Parameters
        ----------
        u: MX
            The independent generalized coordinates
        udot: MX
            The independent generalized velocities
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

        # compute v from u
        # v = self.compute_v_from_u(u)
        v = self.compute_v_from_u_explicit_symbolic(u)
        q = self.q_from_u_and_v(u, v)

        Bvu = self.coupling_matrix(q)
        vdot = Bvu @ udot
        qdot = self.q_from_u_and_v(udot, vdot)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_uu = partitioned_mass_matrix[: self.nb_independent_joints, : self.nb_independent_joints]
        m_uv = partitioned_mass_matrix[: self.nb_independent_joints, self.nb_independent_joints :]
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        modified_mass_matrix = m_uu + m_uv @ Bvu + Bvu.T @ m_vu + Bvu.T @ m_vv @ Bvu
        second_term = m_uv + Bvu.T @ m_vv

        # compute the non linear effect
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_u = non_linear_effect[: self.nb_independent_joints]
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        modified_non_linear_effect = non_linear_effect_u + Bvu.T @ non_linear_effect_v

        # compute the tau
        partitioned_tau = self.partitioned_tau(tau)
        tau_u = partitioned_tau[: self.nb_independent_joints]
        tau_v = partitioned_tau[self.nb_independent_joints :]

        modified_generalized_forces = tau_u + Bvu.T @ tau_v

        uddot = inv(modified_mass_matrix) @ (
            modified_generalized_forces - second_term @ self.biais_vector(q, qdot) - modified_non_linear_effect
        )

        return uddot

    def coupling_matrix(self, q: MX) -> MX:
        """
        Compute the coupling matrix, denoted Bvu in the paper :

        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        """

        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints :]
        Jv_inv = inv(Jv)  # inv_minor otherwise ?

        Ju = J[:, : self.nb_independent_joints]

        return -Jv_inv @ Ju

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Compute the biais vector, denoted b in the paper :

        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.

        """
        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints :]
        Jv_inv = inv(Jv)  # inv_minor otherwise ?

        return - Jv_inv @ self.holonomic_constraints_jacobian(qdot) @ qdot

    def q_from_u_and_v(self, u: MX, v: MX) -> MX:
        """
        Compute the generalized coordinates from the independent and dependent joint coordinates

        Parameters
        ----------
        u: MX
            The independent joint coordinates
        v: MX
            The dependent joint coordinates

        Returns
        -------
        MX
            The generalized coordinates
        """

        q = MX() if isinstance(u, MX) else DM()
        for i in range(self.nb_q):
            if i in self._independent_joint_index:
                q = vertcat(q, u[self._independent_joint_index.index(i)])
            else:
                q = vertcat(q, v[self._dependent_joint_index.index(i)])

        return q

    def compute_v_from_u_explicit_numeric(self, u: MX):
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! numeric version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates of independent joint

        Returns
        -------
        theta:
            The angle of the dependente joint
        """
        model_eigen = biorbd_eigen.Model(self.model.path().absolutePath().to_string())

        index_segment_ref = segment_index(model_eigen, "Arm_location")
        index_forearm = segment_index(model_eigen, "Forearm_location")
        index_marker_hand = marker_index(model_eigen, "CENTER_HAND")
        index_marker_knee = marker_index(model_eigen, "BELOW_KNEE")

        # Find length arm and forearm
        forearm_JCS_trans = model_eigen.segments()[index_forearm].localJCS().trans().to_array()
        hand_JCS_trans = model_eigen.marker(index_marker_hand).to_array()
        l1 = np.sqrt(forearm_JCS_trans[1] ** 2 + forearm_JCS_trans[2] ** 2)
        l2 = np.sqrt(hand_JCS_trans[1] ** 2 + hand_JCS_trans[2] ** 2)

        v = DM.zeros(self.nb_dependent_joints, 1)
        q = self.q_from_u_and_v(u, v)

        # Matrix RT "Arm location" (ref)
        segment_ref_JCS = model_eigen.globalJCS(q.toarray().squeeze(), index_segment_ref).to_array()

        # Perform the forward kinematics
        markers = model_eigen.markers(q.toarray().squeeze())
        marker_knee_in_g = markers[index_marker_knee].to_array()

        # Position markers on arm location frame
        R_arm_global = inv(segment_ref_JCS)
        marker_knee_in_arm = (R_arm_global @ np.concatenate((marker_knee_in_g, np.ones(1)), axis=0))[:3]
        xp = -marker_knee_in_arm[2]
        yp = marker_knee_in_arm[1]

        # Find position dependente joint
        theta = self.inverse_kinematics_2d(
            l1=l1,
            l2=l2,
            xp=xp,
            yp=yp,
        )

        return theta

    def compute_v_from_u_explicit_symbolic(self, u: MX):
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! symbolic version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates of independent joint

        Returns
        -------
        theta:
            The angle of the dependente joint

        """
        index_segment_ref = segment_index(self.model, "Arm_location")
        index_forearm = segment_index(self.model, "Forearm_location")
        index_marker_hand = marker_index(self.model, "CENTER_HAND")
        index_marker_knee = marker_index(self.model, "BELOW_KNEE")

        # Find length arm and forearm
        forearm_JCS_trans = self.model.segments()[index_forearm].localJCS().trans().to_mx()
        hand_JCS_trans = self.model.marker(index_marker_hand).to_mx()
        l1 = cas.sqrt(forearm_JCS_trans[1] ** 2 + forearm_JCS_trans[2] ** 2)
        l2 = cas.sqrt(hand_JCS_trans[1] ** 2 + hand_JCS_trans[2] ** 2)

        v = MX.sym("v", self.nb_dependent_joints)
        q = self.q_from_u_and_v(u, v)

        # Matrix RT "Arm location" (ref)
        segment_ref_JCS = self.model.globalJCS(q, index_segment_ref).to_mx()

        # Perform the forward kinematics
        markers = self.markers(q)
        marker_knee_in_g = markers[index_marker_knee]

        # Position markers on arm location frame
        R_arm_global = inv(segment_ref_JCS)
        marker_knee_in_arm = (R_arm_global @ vertcat(markers[index_marker_knee], cas.MX.ones(1)))[:3]
        xp = -marker_knee_in_arm[2]
        yp = marker_knee_in_arm[1]

        # Find position dependente joint
        theta = self.inverse_kinematics_2d(
            l1=l1,
            l2=l2,
            xp=xp,
            yp=yp,
        )
        return theta

    @staticmethod
    def inverse_kinematics_2d(l1, l2, xp, yp):
        """
        Inverse kinematics with elbow down solution.
        Parameters
        ----------
        l1:
            The length of the arm
        l2:
            The length of the forearm
        xp:
            Coordinate on x of the marker of the knee in the arm's frame
        yp:
            Coordinate on y of the marker of the knee in the arm's frame

        Returns
        -------
        theta:
            The dependent joint
        """

        theta2 = cas.acos((xp**2 + yp**2 - (l2**2 + l1**2)) / (2 * l1 * l2))
        theta1 = cas.atan2(
            (-xp * l2 * cas.sin(theta2) + yp * (l1 + l2 * cas.cos(theta2))),
            (xp * (l1 + l2 * cas.cos(theta2)) + yp * l2 * cas.sin(theta2)),
        )
        return vertcat(theta1, theta2)

    def compute_v_from_u(self, u: MX):
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Symbolic version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates

        Returns
        -------
        MX
            The dependent joint
        """

        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.q_from_u_and_v(u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        residuals = Function(
            "final_states_residuals",
            [decision_variables, u],
            [mx_residuals],
        ).expand()

        opts = {"abstol": 1e-10}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(
            MX(),
            u,
        )

        return v_opt

    def compute_v_from_u_numeric(self, u: DM, v_init=None):
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Numeric version of the function

        Parameters
        ----------
        u: DM
            The generalized coordinates
        v_init: DM
            The initial guess for the dependent joint

        Returns
        -------
        DM
            The numerical values of the dependent joint for a given independent joint u
        """

        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.q_from_u_and_v(u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        residuals = Function(
            "final_states_residuals",
            [decision_variables],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {
            "abstol": 1e-10,
            "print_iteration": True,
            # "constraints": [-1, 1],
        }
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(
            v_init,
        )

        return v_opt


    def compute_vdot(self, q, udot):
        Bvu = self.coupling_matrix(q)
        vdot = Bvu @ udot
        return vdot

    def compute_vddot(self, q, qdot, uddot):
        Bvu = self.coupling_matrix(q)
        biais = self.biais_vector(q, qdot)
        vddot = Bvu @ uddot + biais

        return vddot


    def compute_the_lagrangian_multipliers(
            self, q: MX, qdot: MX, qddot: MX, tau: MX, external_forces: MX = None, f_contacts: MX = None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints:]
        Jvt_inv = inv(Jv.T)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints:, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints:, self.nb_independent_joints:]

        qddot_u = qddot[self._independent_joint_index]
        qddot_v = qddot[self._dependent_joint_index]

        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints:]

        Q = self.partitioned_tau(tau)
        Qv = Q[self.nb_independent_joints:]

        return Jvt_inv @ (m_vu @ qddot_u + m_vv @ qddot_v + non_linear_effect_v - Qv)

    def compute_lagrange_multipliers(self, q, qdot, uddot, tau, f_ext=None, f_contacts=None) -> MX:

        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints:]
        Jv_transpose_inv = inv(transpose(Jv))  # inv_minor otherwise ?

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints:, :self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints:, self.nb_independent_joints:]

        vddot = self.compute_vddot(q, qdot, uddot)

        non_linear_effect = self.model.NonLinearEffect(q, qdot, f_ext=f_ext, f_contacts=f_contacts).to_mx()
        non_linear_effect_v = non_linear_effect[self._dependent_joint_index]

        partitioned_tau = self.partitioned_tau(tau)
        tau_v = partitioned_tau[self.nb_independent_joints:]

        lambdas = Jv_transpose_inv @ (m_vu @ uddot + m_vv @ vddot + non_linear_effect_v - tau_v)

        return lambdas