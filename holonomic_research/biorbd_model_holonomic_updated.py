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
from bioptim import HolonomicBiorbdModel, ConfigureProblem, DynamicsFunctions
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
        l1 = cas.sqrt(forearm_JCS_trans[1] ** 2 + forearm_JCS_trans[2] ** 2)
        l2 = cas.sqrt(hand_JCS_trans[1] ** 2 + hand_JCS_trans[2] ** 2)

        v = MX.sym("v", self.nb_dependent_joints)
        q = self.state_from_partition(u, v)

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
        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven, expand=False)

