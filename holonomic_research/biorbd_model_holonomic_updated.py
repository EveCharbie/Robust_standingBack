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
from casadi import MX, DM, vertcat, horzcat, Function, solve, inv_minor, inv, fmod, pi, transpose
from bioptim import HolonomicBiorbdModel, ConfigureProblem, DynamicsFunctions, SolutionMerge
import numpy as np


class BiorbdModelCustomHolonomic(HolonomicBiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """
    def __init__(self, bio_model: str | biorbd.Model):
        super().__init__(bio_model)

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

        theta2 = cas.acos(
            (xp ** 2 + yp ** 2 - (l2 ** 2 + l1 ** 2)) / (2 * l1 * l2)
        )
        theta1 = cas.atan2(
            (-xp * l2 * cas.sin(theta2) + yp * (l1 + l2 * cas.cos(theta2))),
            (xp * (l1 + l2 * cas.cos(theta2)) + yp * l2 * cas.sin(theta2))
        )
        return vertcat(theta1, theta2)

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
        l1 = cas.sqrt(forearm_JCS_trans[1] ** 2 + forearm_JCS_trans[2] ** 2)    # TODO: Maybe problem with square ?
        l2 = cas.sqrt(hand_JCS_trans[1] ** 2 + hand_JCS_trans[2] ** 2)

        v = MX.sym("v", self.nb_dependent_joints)
        q = self.state_from_partition(u, v)

        # Matrix RT "Arm location" (ref)
        R_arm_global = self.model.globalJCS(q, index_segment_ref).transpose().to_mx()

        # Perform the forward kinematics
        markers = self.markers(q)
        marker_knee_in_g = markers[index_marker_knee]

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
        q = self.state_from_partition(u, v)

        # Matrix RT "Arm location" (ref)
        segment_ref_JCS = model_eigen.globalJCS(q.toarray().squeeze(), index_segment_ref).to_array()

        # Perform the forward kinematics
        markers = model_eigen.markers(q.toarray().squeeze())
        marker_knee_in_g = markers[index_marker_knee].to_array()

        # Position markers on arm location frame
        R_arm_global = inv(segment_ref_JCS)     # TODO: Maybe transpose and not inv ?
        # R_arm_global = transpose(segment_ref_JCS)
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

    @staticmethod
    def holonomic_torque_driven(ocp, nlp, mapping):
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
        names_u = [nlp.model.name_dof[i] for i in mapping["q"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

        name = "qdot_u"
        names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
        names_udot = [names_qdot[i] for i in mapping["qdot"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven) #, expand=False

    def partitioned_forward_dynamics(
        self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")

        # compute q and qdot
        q = self.compute_q(q_u, q_v_init=q_v_init)
        qdot = self.compute_qdot(q, qdot_u)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_uu = partitioned_mass_matrix[: self.nb_independent_joints, : self.nb_independent_joints]
        m_uv = partitioned_mass_matrix[: self.nb_independent_joints, self.nb_independent_joints :]
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        coupling_matrix_vu = self.coupling_matrix(q)
        modified_mass_matrix = (
            m_uu
            + m_uv @ coupling_matrix_vu
            + coupling_matrix_vu.T @ m_vu
            + coupling_matrix_vu.T @ m_vv @ coupling_matrix_vu
        )
        second_term = m_uv + coupling_matrix_vu.T @ m_vv

        # compute the non-linear effect
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_u = non_linear_effect[: self.nb_independent_joints]
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        modified_non_linear_effect = non_linear_effect_u + coupling_matrix_vu.T @ non_linear_effect_v

        # compute the tau
        partitioned_tau = self.partitioned_tau(tau)
        tau_u = partitioned_tau[: self.nb_independent_joints]
        tau_v = partitioned_tau[self.nb_independent_joints :]

        modified_generalized_forces = tau_u + coupling_matrix_vu.T @ tau_v

        qddot_u = inv(modified_mass_matrix) @ (
            modified_generalized_forces - second_term @ self.biais_vector(q, qdot) - modified_non_linear_effect
        )

        return qddot_u

    """""
       def compute_q(self, q_u: MX, q_v_init: MX = None) -> MX:

           Compute the dependent joint from the independent joint
           and integrates them into the variable q

           Parameters
           ----------
           q_u: the q state of the independent joint
           q_v_init

           Returns
           -------
           q: states of the dependent and independent joint


           q_v = self.compute_v_from_u_explicit_symbolic(q_u)
           return self.state_from_partition(q_u, q_v)
       """""

    def compute_q(self, q_u: MX, q_v_init: MX = None) -> MX:
        q_v = self.compute_q_v(q_u, q_v_init)
        return self.state_from_partition(q_u, q_v)

    def compute_qdot_v(self, q: MX, qdot_u: MX) -> MX:
        coupling_matrix_vu = self.coupling_matrix(q)
        return coupling_matrix_vu @ qdot_u

    def compute_qdot(self, q: MX, qdot_u: MX) -> MX:
        qdot_v = self.compute_qdot_v(q, qdot_u)
        return self.state_from_partition(qdot_u, qdot_v)

    def compute_qddot_v(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        coupling_matrix_vu = self.coupling_matrix(q)
        return coupling_matrix_vu @ qddot_u + self.biais_vector(q, qdot)

    def compute_qddot(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        qddot_v = self.compute_qddot_v(q, qdot, qddot_u)
        return self.state_from_partition(qddot_u, qddot_v)

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
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints:]
        partitioned_constraints_jacobian_v_t_inv = inv(partitioned_constraints_jacobian_v.T)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints:, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints:, self.nb_independent_joints:]

        qddot_u = qddot[self._independent_joint_index]
        qddot_v = qddot[self._dependent_joint_index]

        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints:]

        partitioned_tau = self.partitioned_tau(tau)
        partitioned_tau_v = partitioned_tau[self.nb_independent_joints:]

        return partitioned_constraints_jacobian_v_t_inv @ (
                m_vu @ qddot_u + m_vv @ qddot_v + non_linear_effect_v - partitioned_tau_v
        )

    def compute_all_states(self, sol, index_holonomic_model):
        """
        Compute all the states from the solution of the optimal control program

        Parameters
        ----------
        bio_model: HolonomicBiorbdModel
            The biorbd model
        sol:
            The solution of the optimal control program

        Returns
        -------

        """

        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

        n = states[index_holonomic_model]["q_u"].shape[1]
        q = np.zeros((self.nb_q, n))
        qdot = np.zeros((self.nb_q, n))
        qddot = np.zeros((self.nb_q, n))
        lambdas = np.zeros((self.nb_dependent_joints, n))
        tau = np.zeros((self.nb_tau - self.nb_root, n))
        tau_dependent_joint_index = [x - self.nb_root for x in self.dependent_joint_index]
        tau_independent_joint_index = [x - self.nb_root for x in self.independent_joint_index]
        tau_independent_joint_index = [x for x in tau_independent_joint_index if x >= 0]

        for i, independent_joint_index in enumerate(tau_independent_joint_index):
            tau[independent_joint_index, :-1] = controls[index_holonomic_model]["tau"][i, :]
        for i, dependent_joint_index in enumerate(tau_dependent_joint_index):
            tau[dependent_joint_index, :-1] = controls[index_holonomic_model]["tau"][i, :]
        tau_root = np.zeros((self.nb_root, tau.shape[1]))
        tau = np.vstack((tau_root, tau))

        # Partitioned forward dynamics
        q_u_sym = MX.sym("q_u_sym", self.nb_independent_joints, 1)
        qdot_u_sym = MX.sym("qdot_u_sym", self.nb_independent_joints, 1)
        tau_sym = MX.sym("tau_sym", self.nb_tau, 1)
        partitioned_forward_dynamics_func = Function(
            "partitioned_forward_dynamics",
            [q_u_sym, qdot_u_sym, tau_sym],
            [self.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym)],
        )
        # Lagrangian multipliers
        q_sym = MX.sym("q_sym", self.nb_q, 1)
        qdot_sym = MX.sym("qdot_sym", self.nb_q, 1)
        qddot_sym = MX.sym("qddot_sym", self.nb_q, 1)
        compute_lambdas_func = Function(
            "compute_the_lagrangian_multipliers",
            [q_sym, qdot_sym, qddot_sym, tau_sym],
            [self.compute_the_lagrangian_multipliers(q_sym, qdot_sym, qddot_sym, tau_sym)],
        )

        for i in range(n):
            q_v_i = self.compute_q_v(states[index_holonomic_model]["q_u"][:, i]).toarray()
            q[:, i] = self.state_from_partition(states[index_holonomic_model]["q_u"][:, i][:, np.newaxis],
                                                q_v_i).toarray().squeeze()
            qdot[:, i] = self.compute_qdot(q[:, i], states[index_holonomic_model]["qdot_u"][:, i]).toarray().squeeze()
            qddot_u_i = (
                partitioned_forward_dynamics_func(states[index_holonomic_model]["q_u"][:, i],
                                                  states[index_holonomic_model]["qdot_u"][:, i], tau[:, i])
                .toarray()
                .squeeze()
            )
            qddot[:, i] = self.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
            lambdas[:, i] = compute_lambdas_func(q[:, i], qdot[:, i], qddot[:, i], tau[:, i]).toarray().squeeze()

        return q, qdot, qddot, lambdas