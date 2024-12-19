from bioptim import BoundsList
import numpy as np

from .constants import POSE_PROPULSION_START, POSE_LANDING_START, POSE_LANDING_END


def add_x_bounds(bio_models) -> BoundsList:
    """Add bounds to the states of the model q, qdot"""

    n_q = bio_models[0].nb_q
    n_qdot = n_q

    q_bounds = [model.bounds_from_ranges("q") for model in bio_models]
    qdot_bounds = [model.bounds_from_ranges("qdot") for model in bio_models]

    # Phase 0: Propulsion
    q_bounds[0].min[2:7, 0] = np.array(POSE_PROPULSION_START[2:7]) - 0.3
    q_bounds[0].max[2:7, 0] = np.array(POSE_PROPULSION_START[2:7]) + 0.3
    q_bounds[0].max[2, 0] = 0.5
    q_bounds[0].max[5, 0] = 2
    q_bounds[0].min[6, 0] = -2
    q_bounds[0].max[6, 0] = -0.7
    qdot_bounds[0][:, 0] = [0] * n_qdot
    qdot_bounds[0].max[5, :] = 0
    q_bounds[0].min[2, 1:] = -np.pi
    q_bounds[0].max[2, 1:] = np.pi
    q_bounds[0].min[0, :] = -1
    q_bounds[0].max[0, :] = 1
    q_bounds[0].min[1, :] = -1
    q_bounds[0].max[1, :] = 2
    qdot_bounds[0].min[3, :] = 0  # A commenter si marche pas
    q_bounds[0].min[3, 2] = np.pi / 2

    # Phase 1: Flight
    q_bounds[1].min[0, :] = -1
    q_bounds[1].max[0, :] = 1
    q_bounds[1].min[1, :] = -1
    q_bounds[1].max[1, :] = 2
    q_bounds[1].min[2, 0] = -np.pi
    q_bounds[1].max[2, 0] = np.pi * 1.5
    q_bounds[1].min[2, 1] = -np.pi
    q_bounds[1].max[2, 1] = np.pi * 1.5
    q_bounds[1].min[2, -1] = -np.pi
    q_bounds[1].max[2, -1] = np.pi * 1.5

    # Phase 2: Tucked phase
    q_bounds[2].min[0, :] = -1
    q_bounds[2].max[0, :] = 1
    q_bounds[2].min[1, :] = -1
    q_bounds[2].max[1, :] = 2
    q_bounds[2].min[2, 0] = -np.pi
    q_bounds[2].max[2, 0] = np.pi * 1.5
    q_bounds[2].min[2, 1] = -np.pi
    q_bounds[2].max[2, 1] = 2 * np.pi
    q_bounds[2].min[2, 2] = 3 / 4 * np.pi
    q_bounds[2].max[2, 2] = 3 / 2 * np.pi
    q_bounds[2].max[5, :-1] = 2.6
    q_bounds[2].min[5, :-1] = 1.96
    q_bounds[2].max[6, :-1] = -1.5
    q_bounds[2].min[6, :-1] = -2.3

    # Phase 3: Preparation landing
    q_bounds[3].min[0, :] = -1
    q_bounds[3].max[0, :] = 1
    q_bounds[3].min[1, 1:] = -1
    q_bounds[3].max[1, 1:] = 2
    q_bounds[3].min[2, :] = 3 / 4 * np.pi
    q_bounds[3].max[2, :] = 2 * np.pi + 0.5
    q_bounds[3].min[5, -1] = POSE_LANDING_START[5] - 1  # 0.06
    q_bounds[3].max[5, -1] = POSE_LANDING_START[5] + 0.5
    q_bounds[3].min[6, -1] = POSE_LANDING_START[6] - 1
    q_bounds[3].max[6, -1] = POSE_LANDING_START[6] + 0.1

    # Phase 3: Landing
    q_bounds[4].min[5, 0] = POSE_LANDING_START[5] - 1  # 0.06
    q_bounds[4].max[5, 0] = POSE_LANDING_START[5] + 0.5
    q_bounds[4].min[6, 0] = POSE_LANDING_START[6] - 1
    q_bounds[4].max[6, 0] = POSE_LANDING_START[6] + 0.1
    q_bounds[4].min[0, :] = -1
    q_bounds[4].max[0, :] = 1
    q_bounds[4].min[1, :] = -1
    q_bounds[4].max[1, :] = 2
    q_bounds[4].min[2, 0] = 2 / 4 * np.pi
    q_bounds[4].max[2, 0] = 2 * np.pi + 1.66
    q_bounds[4].min[2, 1] = 2 / 4 * np.pi
    q_bounds[4].max[2, 1] = 2 * np.pi + 1.66

    q_bounds[4].max[:, -1] = np.array(POSE_LANDING_END) + 0.2  # 0.5
    q_bounds[4].min[:, -1] = np.array(POSE_LANDING_END) - 0.2

    # large slack on arm angles
    q_bounds[4].max[4, -1] = np.pi / 2
    q_bounds[4].min[4, -1] = 0  # no hyper extension of elbow
    q_bounds[4].max[5, -1] = 2.9
    q_bounds[4].min[5, -1] = 0
    q_bounds[4].min[6, -1] = 0  # no hyper extension of knee

    return q_bounds, qdot_bounds
