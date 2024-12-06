"""
This script plots the actuator functions for each degree of freedom of the model.
"""

from actuators import actuator_function
from actuator_constants import ACTUATORS
import biorbd
import numpy as np
import matplotlib.pyplot as plt


def plot_actuator(ax, tau_max_plus, theta_opt_plus, r_plus, tau_max_minus, theta_opt_minus, r_minus):
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    ax.plot(x, actuator_function(tau_max_minus, theta_opt_minus, r_minus, x), "-r", label="minus")
    ax.plot(x, actuator_function(tau_max_plus, theta_opt_plus, r_plus, x), "-b", label="plus")
    return


def plot_range(ax, min_q, max_q, tau_max_plus, tau_max_minus):
    max_tau = max(tau_max_plus, tau_max_minus)
    ax.plot(np.array([min_q, min_q]), np.array([0, max_tau]), "--k", alpha=0.5)
    ax.plot(np.array([max_q, max_q]), np.array([0, max_tau]), "--k", alpha=0.5)
    return


if __name__ == "__main__":
    model = biorbd.Model("../Model/Pyomecaman_original.bioMod")

    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs = axs.ravel()
    for i_ddl, ddl in enumerate(ACTUATORS.keys()):
        plot_range(
            axs[i_ddl],
            ACTUATORS[ddl].min_q,
            ACTUATORS[ddl].max_q,
            ACTUATORS[ddl].tau_max_plus,
            ACTUATORS[ddl].tau_max_minus,
        )
        plot_actuator(
            axs[i_ddl],
            ACTUATORS[ddl].tau_max_plus,
            ACTUATORS[ddl].theta_opt_plus,
            ACTUATORS[ddl].r_plus,
            ACTUATORS[ddl].tau_max_minus,
            ACTUATORS[ddl].theta_opt_minus,
            ACTUATORS[ddl].r_minus,
        )
        axs[i_ddl].set_title(ddl)
    axs[i_ddl].legend()
    plt.show()
