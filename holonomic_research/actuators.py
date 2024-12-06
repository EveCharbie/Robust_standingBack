import numpy as np


class Joint:
    def __init__(self, tau_max_plus, theta_opt_plus, r_plus, tau_max_minus, theta_opt_minus, r_minus, min_q, max_q):
        self.tau_max_plus = tau_max_plus
        self.theta_opt_plus = theta_opt_plus
        self.r_plus = r_plus
        self.tau_max_minus = tau_max_minus
        self.theta_opt_minus = theta_opt_minus
        self.r_minus = r_minus
        self.min_q = min_q
        self.max_q = max_q


def actuator_function(tau_max, theta_opt, r, x):
    return tau_max * np.exp(-(theta_opt - x)**2 / (2*r**2))