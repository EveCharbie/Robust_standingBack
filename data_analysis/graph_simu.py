import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../holonomic_research/")
from plot_actuators import actuator_function, Joint

# --- Save --- #
def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data_CL.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data_CL: All the data_CL of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp

def time_to_percentage(time):
    time_pourcentage_CL = np.zeros(time.shape)
    for i in range(0, time.shape[0]):
       time_pourcentage_CL[i] = time[i] * 100 / time[time.shape[0]-1]      
    return time_pourcentage_CL

# --- Graph --- #
def graph_all_comparaison(sol_holo, sol_without):
    data_CL = pd.read_pickle(sol_holo)
    lambdas = data_CL["lambda"]
    q_CL_rad = data_CL["q_all"][:, :]
    q_CL_rad_without_last_node = np.hstack((data_CL["q_all"][:, :20], data_CL["q_all"][:, 21:21 + 20],
                                            data_CL["q_all"][:, 21 + 21:21 + 21 + 30],
                                            data_CL["q_all"][:, 21 + 21 + 31:21 + 21 + 31 + 30],
                                            data_CL["q_all"][:, 21 + 21 + 31 + 31:21 + 21 + 31 + 31 + 30]))
    q_CL_rad[6, :] = q_CL_rad[6, :] * -1
    q_CL_deg = np.vstack([q_CL_rad[0:2,:], q_CL_rad[2:, :] *180/np.pi])
    qdot_CL_rad = data_CL["qdot_all"]
    qdot_CL_rad[6, :] = qdot_CL_rad[6, :] * -1
    qdot_CL_deg = np.vstack([qdot_CL_rad[0:2, :], qdot_CL_rad[2:, :] * 180 / np.pi])
    # qdot_CL_deg = qdot_CL_rad
    # qdot_CL_deg = qdot_CL_rad*180/np.pi
    tau_CL = data_CL["tau_all"]
    dof_names = ["Pelvis \n(Translation Y)", "Pelvis \n(Translation Z)",
                 "Pelvis", "Shoulder", "Elbow",
                 "Hip", "Knee", "Ankle"]
    dof_names_tau = ["Shoulder", "Elbow",
                     "Hip", "Knee", "Ankle"]
    time_total_CL = data_CL["time"][-1][-1]

    time_end_phase_CL = []
    time_end_phase_tau_CL = []
    for i in range(len(data_CL["time"])):
        time_end_phase_CL.append(data_CL["time"][i][-1])
        time_end_phase_tau_CL.append(data_CL["time"][i][-2])
    time_end_phase_pourcentage_CL = time_to_percentage(np.vstack(time_end_phase_CL))
    time_end_phase_tau_pourcentage_CL = time_to_percentage(np.vstack(time_end_phase_tau_CL))

    time_CL = np.vstack(data_CL["time"])
    time_pourcentage_CL = time_to_percentage(time_CL)
    time_tau_CL = np.vstack([arr[:-1, :] for arr in data_CL["time"]])
    time_tau_pourcentage_CL = time_to_percentage(time_tau_CL)
    # min_bounds_q = data_CL["min_bounds"]
    # max_bounds_q = data_CL["max_bounds"]

    # Sol 2
    data_without = get_created_data_from_pickle(sol_without)
    q_without_rad = data_without["q_all"][:, :]
    q_without_rad_without_last_node = np.hstack((data_without["q_all"][:, :20], data_without["q_all"][:, 21:21 + 20],
                                                 data_without["q_all"][:, 21 + 21:21 + 21 + 30],
                                                 data_without["q_all"][:, 21 + 21 + 31:21 + 21 + 31 + 30],
                                                 data_without["q_all"][:, 21 + 21 + 31 + 31:21 + 21 + 31 + 31 + 30]))
    q_without_rad[6, :] = q_without_rad[6, :] * -1
    q_without_deg = np.vstack([q_without_rad[0:2,:], q_without_rad[2:, :] *180/np.pi])
    qdot_without_rad = data_without["qdot_all"]
    qdot_without_rad[6, :] = qdot_without_rad[6, :] * -1
    # qdot_without_deg = qdot_without_rad
    qdot_without_deg = np.vstack([qdot_without_rad[0:2, :], qdot_without_rad[2:, :] * 180 / np.pi])
    tau_without = data_without["tau_all"]
    time_total_without = data_without["time"][-1][-1]

    time_end_phase_without = []
    time_end_phase_tau_without = []
    for i in range(len(data_without["time"])):
        time_end_phase_without.append(data_without["time"][i][-1])
        time_end_phase_tau_without.append(data_without["time"][i][-2])
    time_end_phase_pourcentage_without = time_to_percentage(np.vstack(time_end_phase_without))
    time_end_phase_tau_pourcentage_without = time_to_percentage(np.vstack(time_end_phase_tau_without))

    time_without = np.vstack(data_without["time"])  # data_without["time_all"]
    time_pourcentage_without = time_to_percentage(time_without)
    time_tau_without = np.vstack([arr[:-1, :] for arr in data_without["time"]])
    time_tau_pourcentage_without = time_to_percentage(time_tau_without)

    # Figure q
    fig, axs = plt.subplots(3, 3, figsize=(10, 5))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(q_CL_deg[0:2, :]), abs(q_without_deg[0:2, :])])
    y_max_2 = np.max([abs(q_CL_deg[2:5, :]), abs(q_without_deg[2:5, :])])
    y_max_3 = np.max([abs(q_CL_deg[5:, :]), abs(q_without_deg[5:, :])])
    for nb_seg in range(q_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([0, 100]), np.array([0, 0]), '-k', linewidth=0.5)
        axs[num_line, num_col].plot(time_pourcentage_without, q_without_deg[nb_seg], color="tab:blue", label="without \nconstraints",
                                    alpha=0.75, linewidth=1)
        axs[num_line, num_col].plot(time_pourcentage_CL, q_CL_deg[nb_seg], color="tab:orange",
                                    label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase_CL)):
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_without[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_CL[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names[nb_seg], fontsize=8)
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))

        num_col = num_col + 1
        if nb_seg == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

        # Y_label
        axs[0, 0].set_ylabel("Position [m]", fontsize=7)  # Pelvis Translation
        axs[1, 0].set_ylabel(r"F (+) / E (-) [$^\circ$]", fontsize=7)  # Pelvis Rotation
        axs[2, 0].set_ylabel(r"F (+) / E (-) [$^\circ$]", fontsize=7)  # Thight Rotation
        # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Ajouter la légende à la figure de la première ligne, troisième colonne
        axs[0, 2].legend(handles, labels, loc='center', fontsize=8)
        axs[0, 2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=1)
    fig.savefig("q.png", format="png")

    # Figure qdot
    fig, axs = plt.subplots(3, 3, figsize=(10, 5))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(qdot_CL_deg[0:2, :]), abs(qdot_CL_deg[0:2, :])])
    y_max_2 = np.max([abs(qdot_CL_deg[2:5, :]), abs(qdot_CL_deg[2:5, :])])
    y_max_3 = np.max([abs(qdot_CL_deg[5:, :]), abs(qdot_CL_deg[5:, :])])
    for nb_seg in range(qdot_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([0, 100]), np.array([0, 0]), '-k', linewidth=0.5)
        axs[num_line, num_col].plot(time_pourcentage_without, qdot_without_deg[nb_seg], color="tab:blue",
                                    label="without \nconstraints", alpha=0.75, linewidth=1)
        axs[num_line, num_col].plot(time_pourcentage_CL, qdot_CL_deg[nb_seg], color="tab:orange",
                                    label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase_CL)):
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_without[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_CL[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names[nb_seg], fontsize=8)

        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)

        num_col = num_col + 1
        if nb_seg == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

        # Y_label
        axs[0, 0].set_ylabel("Velocity [m/s]", fontsize=7)  # Pelvis Translation
        axs[0, 1].set_yticklabels([])  # Pelvis Translation
        axs[1, 0].set_ylabel("F (+) / E (-) [" + r"$^\circ$/s" + "]", fontsize=7)  # Pelvis Rotation
        axs[1, 1].set_yticklabels([])  # Arm Rotation
        axs[1, 2].set_yticklabels([])  # Forearm Rotation
        axs[2, 0].set_ylabel("F (+) / E (-) [" + r"$^\circ$/s" + "]", fontsize=7)  # Thight Rotation
        axs[2, 1].set_yticklabels([])  # Leg Rotation
        axs[2, 2].set_yticklabels([])  # Foot Rotation
        # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Ajouter la légende à la figure de la première ligne, troisième colonne
        axs[0, 2].legend(handles, labels, loc='center', fontsize=7)
        axs[0, 2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    fig.savefig("qdot.png", format="png")


    # Theoretical min and max bound on tau based on actuators
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

    tau_CL_min_bound = np.zeros((5, tau_CL.shape[1]))
    tau_CL_max_bound = np.zeros((5, tau_CL.shape[1]))
    tau_without_min_bound = np.zeros((5, tau_without.shape[1]))
    tau_without_max_bound = np.zeros((5, tau_without.shape[1]))
    for nb_seg, key in enumerate(actuators.keys()):
        tau_CL_min_bound[nb_seg, :] = -actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q_CL_rad_without_last_node[nb_seg+3])
        tau_CL_max_bound[nb_seg, :] = actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q_CL_rad_without_last_node[nb_seg+3])
        tau_without_min_bound[nb_seg, :] = -actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q_without_rad_without_last_node[nb_seg+3])
        tau_without_max_bound[nb_seg, :] = actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q_without_rad_without_last_node[nb_seg+3])

    # Figure tau
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    num_col = 1 
    num_line = 0

    y_max_1 = np.max([abs(tau_without[0:2, :]), abs(tau_CL[0:2, :])])
    y_max_2 = np.max([abs(tau_without[2:, :]), abs(tau_CL[2:, :])])

    axs[0, 0].plot([], [], color="tab:orange", label="with holonomics \nconstraints")
    axs[0, 0].plot([], [], color="tab:blue", label="without \nconstraints")
    axs[0, 0].fill_between([], [], [], color="tab:orange", alpha=0.1, label="physiological \nmin/max" + r" $\tau$ with", linewidth=0.5)
    axs[0, 0].fill_between([], [], [], color="tab:blue", alpha=0.1, label="physiological \nmin/max" + r" $\tau$ without", linewidth=0.5)
    axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
    axs[0, 0].axis('off')

    for nb_seg in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([0, 100]), np.array([0, 0]), '-k', linewidth=0.5)

        # axs[num_line, num_col].step(range(len(tau_without[nb_seg])), tau_without_max_bound[nb_seg], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(range(len(tau_without[nb_seg])), tau_without_max_bound[nb_seg], np.ones(tau_without_max_bound[nb_seg].shape) * 1000, color="tab:blue", alpha=0.1, step='pre', linewidth=0.5)
        # axs[num_line, num_col].step(range(len(tau_without[nb_seg])), tau_without_min_bound[nb_seg], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(range(len(tau_without[nb_seg])), np.ones(tau_without_max_bound[nb_seg].shape) * -1000, tau_without_min_bound[nb_seg], color="tab:blue", alpha=0.1, step='pre', linewidth=0.5)
        # axs[num_line, num_col].step(range(len(tau_CL[nb_seg])), tau_CL_max_bound[nb_seg], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(range(len(tau_CL[nb_seg])), tau_CL_max_bound[nb_seg], np.ones(tau_without_max_bound[nb_seg].shape) * 1000, color="tab:orange", alpha=0.1, step='pre', linewidth=0.5)
        # axs[num_line, num_col].step(range(len(tau_CL[nb_seg])), tau_CL_min_bound[nb_seg], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].fill_between(range(len(tau_CL[nb_seg])), np.ones(tau_without_max_bound[nb_seg].shape) * -1000, tau_CL_min_bound[nb_seg], color="tab:orange", alpha=0.1, step='pre', linewidth=0.5)

        axs[num_line, num_col].step(range(len(tau_without[nb_seg])), tau_without[nb_seg], color="tab:blue", alpha=0.75, linewidth=1, label="without \nconstraints", where='mid')
        axs[num_line, num_col].step(range(len(tau_CL[nb_seg])), tau_CL[nb_seg], color="tab:orange", alpha=0.75, linewidth=1, label="with holonomics \nconstraints", where='mid')

        for xline in range(len(time_end_phase_CL)):
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_CL[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_pourcentage_without[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque [Nm]", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Joint torque [Nm]", fontsize=7)  # Leg Rotation
    axs[0, 2].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 2].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig("tau.png", format="png")

    # Figure lambdas
    time_tuck = data_CL["time"][2][:-1] - data_CL["time"][2][0]
    fig = plt.figure()
    plt.plot(time_tuck, lambdas[0], color='r', label=["Normal force"])
    plt.plot(time_tuck, lambdas[1], color='g', label=["Shear force"])
    plt.ylabel("Force on the tibia [N]")
    plt.xlabel("Time [s]")
    plt.legend()
    fig.savefig("lambdas.png", format="png")


    # Tau ratio only CL phase
    tau_CL_ratio = np.zeros(data_CL["tau"][2].shape)
    tau_without_ratio = np.zeros(data_without["tau"][2].shape)
    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    for i_dof in range(5):
        axs[0, 0].step(time_tuck, data_CL["tau"][2][i_dof, :], color="tab:orange", label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        axs[0, 0].step(time_tuck, data_without["tau"][2][i_dof, :], color="tab:blue", label="without \nconstraints", alpha=0.75, linewidth=1)

        for i_node in range(data_without["tau"][2].shape[1]):
            if data_CL["tau"][2][i_dof, i_node] > 0:
                tau_CL_ratio[i_dof, i_node] = data_CL["tau"][2][i_dof, i_node] / tau_CL_max_bound[i_dof, 40+i_node]
            else:
                tau_CL_ratio[i_dof, i_node] = np.abs(data_CL["tau"][2][i_dof, i_node] / tau_CL_min_bound[i_dof, 40+i_node])
            if data_without["tau"][2][i_dof, i_node] > 0:
                tau_without_ratio[i_dof, i_node] = data_without["tau"][2][i_dof, i_node] / tau_without_max_bound[i_dof, 40+i_node]
            else:
                tau_without_ratio[i_dof, i_node] = np.abs(data_without["tau"][2][i_dof, i_node] / tau_without_min_bound[i_dof, 40+i_node])
        axs[1, 0].step(time_tuck, tau_CL_ratio[i_dof, :], color="tab:orange",
                       label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        axs[1, 0].step(time_tuck, tau_without_ratio[i_dof, :], color="tab:blue",
                       label="without \nconstraints", alpha=0.75, linewidth=1)

    sum_tau_CL = np.sum(np.abs(data_CL["tau"][2]))  # @mickaelbegon: Should we normalize by the phase duration?
    sum_tau_without = np.sum(np.abs(data_without["tau"][2]))
    axs[0, 1].bar([0, 1], [sum_tau_CL, sum_tau_without], color=["tab:orange", "tab:blue"])
    # axs[0, 1].get_xaxis().set_visible(False)
    sum_tau_ratio_CL = np.sum(tau_CL_ratio)  # @mickaelbegon: Should we normalize by the phase duration?
    sum_tau_ratio_without = np.sum(tau_without_ratio)
    axs[1, 1].bar([0, 1], [sum_tau_ratio_CL, sum_tau_ratio_without], color=["tab:orange", "tab:blue"])
    # axs[1, 1].get_xaxis().set_visible(False)

    axs[0, 0].set_xlabel("Tucked time [s]")
    axs[0, 0].set_ylabel("Joint torque [Nm]")
    axs[1, 0].set_xlabel("Tucked time [s]")
    axs[1, 0].set_ylabel("Physiological joint torque ratio")
    axs[0, 1].set_xticks([0, 1], ["with", "without"])
    axs[1, 1].set_xticks([0, 1], ["with", "without"])

    axs[1, 1].plot([], [], color="tab:orange", label=r"$\tau$" + " with holonomics \nconstraints")
    axs[1, 1].plot([], [], color="tab:blue", label=r"$\tau$" + " without \nconstraints")
    axs[1, 1].fill_between([], [], [], color="tab:orange", label="sum " + r"$\tau$" + " with \nholonomics constrain")
    axs[1, 1].fill_between([], [], [], color="tab:blue", label="sum " + r"$\tau$" + " without \nholonomics constrain")
    axs[1, 1].legend(loc='center right', bbox_to_anchor=(1.85, 1.5), fontsize=8)
    plt.subplots_adjust(right=0.75, hspace=0.25)
    plt.savefig("tau_ratio_tucked_phase.png", format="png")


    # Tau ratio all phases
    dt_CL = np.vstack((data_CL["time"][0][1:] - data_CL["time"][0][:-1],
                       data_CL["time"][1][1:] - data_CL["time"][1][:-1],
                       data_CL["time"][2][1:] - data_CL["time"][2][:-1],
                       data_CL["time"][3][1:] - data_CL["time"][3][:-1],
                       data_CL["time"][4][1:] - data_CL["time"][4][:-1]))
    dt_without = np.vstack((data_without["time"][0][1:] - data_without["time"][0][:-1],
                            data_without["time"][1][1:] - data_without["time"][1][:-1],
                            data_without["time"][2][1:] - data_without["time"][2][:-1],
                            data_without["time"][3][1:] - data_without["time"][3][:-1],
                            data_without["time"][4][1:] - data_without["time"][4][:-1]))
    tau_CL_ratio_all = np.zeros(tau_CL.shape)
    tau_without_ratio_all = np.zeros(tau_without.shape)
    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    for i_dof in range(5):
        axs[0, 0].step(time_tau_pourcentage_CL, tau_CL[i_dof, :], color="tab:orange", label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        axs[0, 0].step(time_tau_pourcentage_without, tau_without[i_dof, :], color="tab:blue", label="without \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase_CL)):
            axs[0, 0].axvline(time_end_phase_pourcentage_CL[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
            axs[0, 0].axvline(time_end_phase_pourcentage_without[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)

        for i_node in range(tau_CL.shape[1]):
            if tau_CL[i_dof, i_node] > 0:
                tau_CL_ratio_all[i_dof, i_node] = tau_CL[i_dof, i_node] / tau_CL_max_bound[i_dof, i_node]
            else:
                tau_CL_ratio_all[i_dof, i_node] = np.abs(tau_CL[i_dof, i_node] / tau_CL_min_bound[i_dof, i_node])
            if tau_without[i_dof, i_node] > 0:
                tau_without_ratio_all[i_dof, i_node] = tau_without[i_dof, i_node] / tau_without_max_bound[i_dof, i_node]
            else:
                tau_without_ratio_all[i_dof, i_node] = np.abs(tau_without[i_dof, i_node] / tau_without_min_bound[i_dof, i_node])
        axs[1, 0].step(time_tau_pourcentage_CL, tau_CL_ratio_all[i_dof, :], color="tab:orange",
                       label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        axs[1, 0].step(time_tau_pourcentage_without, tau_without_ratio_all[i_dof, :], color="tab:blue",
                       label="without \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase_CL)):
            axs[1, 0].axvline(time_end_phase_pourcentage_CL[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
            axs[1, 0].axvline(time_end_phase_pourcentage_without[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
    sum_tau_all_CL = np.sum(np.abs(tau_CL * dt_CL.T))
    sum_tau_all_without = np.sum(np.abs(tau_without * dt_without.T))
    axs[0, 1].bar([0, 1], [sum_tau_all_CL, sum_tau_all_without], color=["tab:orange", "tab:blue"])
    # axs[0, 1].get_xaxis().set_visible(False)
    sum_tau_all_ratio_CL = np.sum(np.abs(tau_CL_ratio_all * dt_CL.T))
    sum_tau_all_ratio_without = np.sum(np.abs(tau_without_ratio_all * dt_without.T))
    axs[1, 1].bar([0, 1], [sum_tau_all_ratio_CL, sum_tau_all_ratio_without], color=["tab:orange", "tab:blue"])
    # axs[1, 1].get_xaxis().set_visible(False)

    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Joint torque [Nm]")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Physiological joint torque ratio")
    axs[0, 1].set_xticks([0, 1], ["with", "without"])
    axs[1, 1].set_xticks([0, 1], ["with", "without"])

    axs[1, 1].plot([], [], color="tab:orange", label=r"$\tau$" + " with holonomics \nconstraints")
    axs[1, 1].plot([], [], color="tab:blue", label=r"$\tau$" + " without \nconstraints")
    axs[1, 1].fill_between([], [], [], color="tab:orange", label="sum " + r"$\tau$" + " with \nholonomics constrain")
    axs[1, 1].fill_between([], [], [], color="tab:blue", label="sum " + r"$\tau$" + " without \nholonomics constrain")
    axs[1, 1].legend(loc='center right', bbox_to_anchor=(1.85, 1.5), fontsize=8)
    plt.subplots_adjust(right=0.75, hspace=0.25)

    plt.savefig("tau_ratio_all.png", format="png")
    plt.show()
