import numpy as np
from .actuators import Joint

TAU_MAX = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
TAU_MIN = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]


def initialize_tau(coef=0.7):
    tau_min = [i * coef for i in TAU_MIN]
    tau_max = [i * coef for i in TAU_MAX]
    tau_init = 0
    return tau_min, tau_max, tau_init


ACTUATORS = {
    "Shoulders": Joint(
        tau_max_plus=112.8107 * 2,
        theta_opt_plus=41.0307 * np.pi / 180,
        r_plus=109.6679 * np.pi / 180,
        tau_max_minus=162.7655 * 2,
        theta_opt_minus=101.6627 * np.pi / 180,
        r_minus=103.9095 * np.pi / 180,
        min_q=-0.7,
        max_q=3.1,
    ),
    # this one was not measured, I just tried to fit
    # https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
    "Elbows": Joint(
        tau_max_plus=80 * 2,
        theta_opt_plus=np.pi / 2 - 0.1,
        r_plus=40 * np.pi / 180,
        tau_max_minus=50 * 2,
        theta_opt_minus=np.pi / 2 - 0.1,
        r_minus=70 * np.pi / 180,
        min_q=0,
        max_q=2.09,
    ),
    "Hips": Joint(
        tau_max_plus=220.3831 * 2,
        theta_opt_plus=25.6939 * np.pi / 180,
        r_plus=56.4021 * np.pi / 180,
        tau_max_minus=490.5938 * 2,
        theta_opt_minus=72.5836 * np.pi / 180,
        r_minus=48.6999 * np.pi / 180,
        min_q=-0.4,
        max_q=2.6,
    ),
    "Knees": Joint(
        tau_max_plus=367.6643 * 2,
        theta_opt_plus=-61.7303 * np.pi / 180,
        r_plus=31.7218 * np.pi / 180,
        tau_max_minus=177.9694 * 2,
        theta_opt_minus=-33.2908 * np.pi / 180,
        r_minus=57.0370 * np.pi / 180,
        min_q=-2.3,
        max_q=0.02,
    ),
    "Ankles": Joint(
        tau_max_plus=153.8230 * 2,
        theta_opt_plus=0.7442 * np.pi / 180,
        r_plus=58.9832 * np.pi / 180,
        tau_max_minus=171.9903 * 2,
        theta_opt_minus=12.6824 * np.pi / 180,
        r_minus=21.8717 * np.pi / 180,
        min_q=-0.7,
        max_q=0.7,
    ),
}
