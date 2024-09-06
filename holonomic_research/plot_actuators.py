
import biorbd
import numpy as np
import matplotlib.pyplot as plt


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

def plot_actuator(ax, tau_max_plus, theta_opt_plus, r_plus, tau_max_minus, theta_opt_minus, r_minus):
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    ax.plot(x, actuator_function(tau_max_minus, theta_opt_minus, r_minus, x), '-r', label='minus')
    ax.plot(x, actuator_function(tau_max_plus, theta_opt_plus, r_plus, x), '-b', label="plus")
    return

def plot_range(ax, min_q, max_q, tau_max_plus, tau_max_minus):
    max_tau = max(tau_max_plus, tau_max_minus)
    ax.plot(np.array([min_q, min_q]), np.array([0, max_tau]), '--k', alpha=0.5)
    ax.plot(np.array([max_q, max_q]), np.array([0, max_tau]), '--k', alpha=0.5)
    return

if __name__ == "__main__":
    model = biorbd.Model("../Model/Pyomecaman_original.bioMod")

    actuators = {"Shoulders": Joint(tau_max_plus=112.8107*2,
                                    theta_opt_plus=-41.0307*np.pi/180,
                                    r_plus=109.6679*np.pi/180,
                                    tau_max_minus=162.7655*2,
                                    theta_opt_minus=-101.6627*np.pi/180,
                                    r_minus=103.9095*np.pi/180,
                                    min_q=-0.7,
                                    max_q=3.1),
                "Elbows": Joint(tau_max_plus=100*2,
                                theta_opt_plus=np.pi/2-0.1,
                                r_plus=40*np.pi/180,
                                tau_max_minus=50*2,
                                theta_opt_minus=np.pi/2-0.1,
                                r_minus=70*np.pi/180,
                                min_q=0,
                                max_q=2.09),  # this one was not measured, I just tried to fit https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
                "Hips": Joint(tau_max_plus=220.3831*2,
                            theta_opt_plus=25.6939*np.pi/180,
                            r_plus=56.4021*np.pi/180,
                            tau_max_minus=490.5938*2,
                            theta_opt_minus=72.5836*np.pi/180,
                            r_minus=48.6999*np.pi/180,
                            min_q=-0.4,
                            max_q=2.6),
                "Knees": Joint(tau_max_plus=367.6643*2,
                            theta_opt_plus=-61.7303*np.pi/180,
                            r_plus=31.7218*np.pi/180,
                            tau_max_minus=177.9694*2,
                            theta_opt_minus=-33.2908*np.pi/180,
                            r_minus=57.0370*np.pi/180,
                            min_q=-2.3,
                            max_q=0.02),
                "Ankles": Joint(tau_max_plus=153.8230*2,
                            theta_opt_plus=0.7442*np.pi/180,
                            r_plus=58.9832*np.pi/180,
                            tau_max_minus=171.9903*2,
                            theta_opt_minus=12.6824*np.pi/180,
                            r_minus=21.8717*np.pi/180,
                            min_q=-0.7,
                            max_q=0.7)
            }

    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs = axs.ravel()
    for i_ddl, ddl in enumerate(actuators.keys()):
        plot_range(axs[i_ddl], actuators[ddl].min_q, actuators[ddl].max_q, actuators[ddl].tau_max_plus, actuators[ddl].tau_max_minus)
        plot_actuator(axs[i_ddl], actuators[ddl].tau_max_plus, actuators[ddl].theta_opt_plus, actuators[ddl].r_plus, actuators[ddl].tau_max_minus, actuators[ddl].theta_opt_minus, actuators[ddl].r_minus)
        axs[i_ddl].set_title(ddl)
    axs[i_ddl].legend()
    plt.show()



