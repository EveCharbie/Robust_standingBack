import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from Create_cylinder_insole import distance_between_line_sensors

# --- Functions --- #
def lissage(signal_brut, L):
    """

    :param signal_brut: signal
    :param L: step for smoothing
    :return: signal smoothing
    """
    res = np.copy(signal_brut)  # duplication des valeurs
    for i in range(1, len(signal_brut) - 1):  # toutes les valeurs sauf la première et la dernière
        L_g = min(i, L)  # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut) - i - 1, L)  # nombre de valeurs disponibles à droite
        Li = min(L_g, L_d)
        res[i] = np.sum(signal_brut[i - Li : i + Li + 1]) / (2 * Li + 1)
    return res


def distance_activation_sensor(coordonnes_insoles, position_activation, name_activation):
    distance = {
        "value": abs(position_activation[:, np.newaxis] - coordonnes_insoles),
        "activation_name": name_activation,
    }

    return distance


def find_activation_sensor(distance_sensor_y, position_activation_y):
    index_post_L = []
    index_pre_L = []
    for valeur_proche in position_activation_y:
        differences = [abs(x - valeur_proche) for x in distance_sensor_y]
        index_proche_1_L = min(range(len(differences)), key=differences.__getitem__)
        index_pre_L.append(index_proche_1_L)
        differences[index_proche_1_L] = float("inf")  # Exclure la première différence minimale
        index_proche_2_L = min(range(len(differences)), key=differences.__getitem__)
        index_post_L.append(index_proche_2_L)

    pourcentage_L = []
    for i in range(len(index_post_L)):
        pourcentage_L.append(
            abs(distance_sensor_y[index_pre_L[i]] - position_activation_y[i])
            / abs(distance_sensor_y[index_pre_L[i]] - distance_sensor_y[index_post_L[i]])
        )

    return np.array([index_pre_L, index_post_L, pourcentage_L])


# --- Cartography insoles --- #
def cartography_insole(file_insole, file_info_insole, fig_name: str, FLAG_PLOT=False):

    sensor_45 = file_insole.columns[
        1:-1
    ].to_list()  #  List sensor on the insole size 45 (without columns Sync and Time)
    coordonnees_insole_45 = file_info_insole.loc[0:7, sensor_45]

    # Plot : Insoles
    if FLAG_PLOT:
        fig, axs = plt.subplots(2)
        fig.suptitle("Insoles cartography")

        # Subplot 1 : insoles R
        axs[0].plot(coordonnees_insole_45.iloc[7, :], coordonnees_insole_45.iloc[6, :], "ro")
        axs[0].set_ylabel("Position Y (mm)", fontsize=14)
        axs[0].title.set_text("Insole Right")

        # Subplot 2 : insoles L
        axs[1].plot(coordonnees_insole_45.iloc[5, :], coordonnees_insole_45.iloc[4, :], "ro")
        axs[1].set_xlabel("Position X (mm)", fontsize=14)
        axs[1].set_ylabel("Position Y (mm)", fontsize=14)
        axs[1].title.set_text("Insole Left")
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()


# Find all the activation
# Load insoles
def position_activation(file_insole_R, file_insole_L, file_info_insole, FLAG_PLOT=False):
    """

    :param file_insole_R: File of the right insole
    :param file_insole_L: File of the left insole
    :param file_info_insole: File with the information of the insole (coordinate sensor insole)
    :param FLAG_PLOT: if you want the plot related to the detection of the activation position on the insole
    :return: The position of activation for the right and the left insoles, and the distance between a reference activation and all the sensor
    """

    # Baseline activation
    mean_sensor_R_baseline = file_insole_R.iloc[200:1500, :].mean(axis=0)
    mean_sensor_L_baseline = file_insole_L.iloc[200:1500, :].mean(axis=0)
    markers_insole_R_baseline = file_insole_R.iloc[:, :] - mean_sensor_R_baseline
    markers_insole_L_baseline = file_insole_L.iloc[:, :] - mean_sensor_L_baseline

    # Find sensors who were activated
    sensor_min_max_R = pd.DataFrame(
        [markers_insole_R_baseline.max(), markers_insole_R_baseline.min()],
        index=[0, 1],
        columns=markers_insole_R_baseline.columns.tolist(),
    )
    sensor_min_max_L = pd.DataFrame(
        [markers_insole_L_baseline.max(), markers_insole_L_baseline.min()],
        index=[0, 1],
        columns=markers_insole_L_baseline.columns.tolist(),
    )

    filtered_sensor_min_max_R = sensor_min_max_R.loc[
        :, (sensor_min_max_R.iloc[0] > 1.5) & (sensor_min_max_R.iloc[1] > -0.3)
    ]
    filtered_sensor_min_max_L = sensor_min_max_L.loc[
        :, (sensor_min_max_L.iloc[0] > 1.5) & (sensor_min_max_L.iloc[1] >= -0.3)
    ]

    # Total activation
    markers_insole_R_baseline_total = markers_insole_R_baseline.loc[:, filtered_sensor_min_max_R.columns.to_list()].sum(
        axis=1
    )
    markers_insole_L_baseline_total = markers_insole_L_baseline.loc[:, filtered_sensor_min_max_L.columns.to_list()].sum(
        axis=1
    )

    peaks_L, properties_L = signal.find_peaks(
        markers_insole_L_baseline_total, distance=400, plateau_size=1, height=1.5, width=20
    )
    peaks_R, properties_R = signal.find_peaks(
        markers_insole_R_baseline_total, distance=300, plateau_size=1, height=1.5, width=20
    )

    if FLAG_PLOT:
        fig, axes = plt.subplots(nrows=2, ncols=1)
        fig.suptitle(" Totale activation insole right and left")
        axes[0].plot(markers_insole_R_baseline_total, color="green")
        axes[0].plot(peaks_R, markers_insole_R_baseline_total[peaks_R], "x", color="red", label="Peak insole right")
        axes[0].set_ylabel("Pressure (N/cm2)", fontsize=14)
        axes[0].legend(loc="upper left")
        axes[0].set_title("Insole right")

        axes[1].plot(markers_insole_L_baseline_total)
        axes[1].plot(peaks_L, markers_insole_L_baseline_total[peaks_L], "x", label="Peak insole left")
        axes[1].set_ylabel("Pressure (N/cm2)", fontsize=14)
        axes[1].set_xlabel("Time", fontsize=14)
        axes[1].legend(loc="upper left")
        axes[1].set_title("Insole left")
        fig.tight_layout()
        plt.savefig("Figures/insoles_totale_activation_markers.svg")
        fig.clf()

    position_activation_R = np.zeros(shape=(2, peaks_R.shape[0]))
    position_activation_L = np.zeros(shape=(2, peaks_L.shape[0]))
    position_activation_R_ref = np.zeros(shape=(2, peaks_R.shape[0]))
    position_activation_L_ref = np.zeros(shape=(2, peaks_L.shape[0]))

    for i in range(peaks_R.shape[0]):
        # Take 10 frames before and after the peak and find sensors activated
        insole_R_activate_peak = markers_insole_R_baseline.loc[:, filtered_sensor_min_max_R.columns.to_list()][
            peaks_R[i] - 10 : peaks_R[i] + 10
        ]
        # Sélection des colonnes où les valeurs sont supérieures à zéro
        colonnes_sup_zero_R = insole_R_activate_peak.loc[:, (insole_R_activate_peak > 0).any()]
        coordonnees_R = file_info_insole.loc[:, colonnes_sup_zero_R.columns.to_list()]
        poids_R = np.mean(colonnes_sup_zero_R, axis=0)
        position_activation_R[0, i] = np.average(coordonnees_R.iloc[6, :], axis=0, weights=poids_R)  #   Position x
        position_activation_R[1, i] = np.average(coordonnees_R.iloc[7, :], axis=0, weights=poids_R)  #   Position y
        position_activation_R_ref[:, i] = file_info_insole.iloc[6:8, 0] - position_activation_R[:, i]

    for i in range(peaks_L.shape[0]):
        insole_L_activate_peak = markers_insole_L_baseline.loc[:, filtered_sensor_min_max_L.columns.to_list()][
            peaks_L[i] - 10 : peaks_L[i] + 10
        ]
        colonnes_sup_zero_L = insole_L_activate_peak.loc[:, (insole_L_activate_peak > 0).any()]
        coordonnees_L = file_info_insole.loc[:, colonnes_sup_zero_L.columns.to_list()]
        poids_L = np.mean(colonnes_sup_zero_L, axis=0)

        position_activation_L[0, i] = np.average(coordonnees_L.iloc[4, :], axis=0, weights=poids_L)  #   Position x
        position_activation_L[1, i] = np.average(coordonnees_L.iloc[5, :], axis=0, weights=poids_L)  #   Position y
        position_activation_L_ref[:, i] = file_info_insole.iloc[6:8, 0] - position_activation_L[:, i]

    distance_sensor_y = distance_between_line_sensors(file_info_insole)

    activation_R = find_activation_sensor(distance_sensor_y, position_activation_R_ref[1, :])
    activation_L = find_activation_sensor(distance_sensor_y, position_activation_L_ref[1, :])

    # Plot position markers semelles
    if FLAG_PLOT:
        fig, axs = plt.subplots(2)
        fig.suptitle("Insoles cartography")

        # Subplot 1 : insoles R
        axs[0].plot(file_info_insole.iloc[7, :], file_info_insole.iloc[6, :], "ro")
        axs[0].plot(position_activation_R[1, :], position_activation_R[0, :], "bo")
        axs[0].set_ylabel("Position Y (mm)", fontsize=14)
        axs[0].title.set_text("Insole Right")

        # Subplot 2 : insoles L
        axs[1].plot(file_info_insole.iloc[5, :], file_info_insole.iloc[4, :], "ro")
        axs[1].plot(position_activation_L[1, :], position_activation_L[0, :], "bo")
        axs[1].set_xlabel("Position X (mm)", fontsize=14)
        axs[1].set_ylabel("Position Y (mm)", fontsize=14)
        axs[1].title.set_text("Insole Left")
        fig.tight_layout()
        plt.savefig("Figures/insoles_position_markers.svg")
        fig.clf()

    if FLAG_PLOT:

        # Subplot 1 : insoles R
        plt.plot(file_info_insole.iloc[7, :], file_info_insole.iloc[6, :], "ro")
        plt.plot(
            position_activation_R[1, 0],
            position_activation_R[0, 0],
            marker="o",
            markersize=20,
            markerfacecolor="green",
            label="Activation_1",
        )
        plt.plot(
            position_activation_R[1, 1],
            position_activation_R[0, 1],
            marker="o",
            markersize=20,
            markerfacecolor="blue",
            label="Activation_2",
        )
        plt.plot(
            position_activation_R[1, 2],
            position_activation_R[0, 2],
            marker="o",
            markersize=20,
            markerfacecolor="orange",
            label="Activation_3",
        )
        plt.plot(
            position_activation_R[1, 3],
            position_activation_R[0, 3],
            marker="o",
            markersize=20,
            markerfacecolor="red",
            label="Activation_4",
        )
        plt.plot(
            position_activation_R[1, 4],
            position_activation_R[0, 4],
            marker="o",
            markersize=20,
            markerfacecolor="brown",
            label="Activation_5",
        )
        plt.plot(
            position_activation_R[1, 5],
            position_activation_R[0, 5],
            marker="o",
            markersize=20,
            markerfacecolor="pink",
            label="Activation_6",
        )
        plt.plot(
            position_activation_R[1, 6],
            position_activation_R[0, 6],
            marker="o",
            markersize=20,
            markerfacecolor="gray",
            label="Activation_7",
        )
        plt.plot(
            position_activation_R[1, 7],
            position_activation_R[0, 7],
            marker="o",
            markersize=20,
            markerfacecolor="olive",
            label="Activation_8",
        )
        plt.plot(
            position_activation_R[1, 8],
            position_activation_R[0, 8],
            marker="o",
            markersize=20,
            markerfacecolor="purple",
            label="Activation_9",
        )
        plt.plot(
            position_activation_R[1, 9],
            position_activation_R[0, 9],
            marker="o",
            markersize=20,
            markerfacecolor="cyan",
            label="Activation_10",
        )
        plt.ylabel("Position Y (mm)", fontsize=14)
        plt.xlabel("Position X (mm)", fontsize=14)
        plt.legend()
        plt.savefig("Figures/insoles_position_activation_R.svg")
        plt.clf()

        plt.plot(file_info_insole.iloc[5, :], file_info_insole.iloc[4, :], "ro")
        plt.plot(
            position_activation_L[1, 0],
            position_activation_L[0, 0],
            marker="o",
            markersize=20,
            markerfacecolor="green",
            label="Activation_1",
        )
        plt.plot(
            position_activation_L[1, 1],
            position_activation_L[0, 1],
            marker="o",
            markersize=20,
            markerfacecolor="blue",
            label="Activation_2",
        )
        plt.plot(
            position_activation_L[1, 2],
            position_activation_L[0, 2],
            marker="o",
            markersize=20,
            markerfacecolor="orange",
            label="Activation_3",
        )
        plt.plot(
            position_activation_L[1, 3],
            position_activation_L[0, 3],
            marker="o",
            markersize=20,
            markerfacecolor="red",
            label="Activation_4",
        )
        plt.plot(
            position_activation_L[1, 4],
            position_activation_L[0, 4],
            marker="o",
            markersize=20,
            markerfacecolor="brown",
            label="Activation_4",
        )
        plt.plot(
            position_activation_L[1, 5],
            position_activation_L[0, 5],
            marker="o",
            markersize=20,
            markerfacecolor="pink",
            label="Activation_5",
        )
        plt.plot(
            position_activation_L[1, 6],
            position_activation_L[0, 6],
            marker="o",
            markersize=20,
            markerfacecolor="gray",
            label="Activation_6",
        )
        plt.plot(
            position_activation_L[1, 7],
            position_activation_L[0, 7],
            marker="o",
            markersize=20,
            markerfacecolor="olive",
            label="Activation_7",
        )
        plt.plot(
            position_activation_L[1, 8],
            position_activation_L[0, 8],
            marker="o",
            markersize=20,
            markerfacecolor="purple",
            label="Activation_8",
        )
        plt.plot(
            position_activation_L[1, 9],
            position_activation_L[0, 9],
            marker="o",
            markersize=20,
            markerfacecolor="cyan",
            label="Activation_9",
        )
        plt.plot(
            position_activation_L[1, 10],
            position_activation_L[0, 10],
            marker="o",
            markersize=20,
            markerfacecolor="black",
            label="Activation_10",
        )
        plt.ylabel("Position Y (mm)", fontsize=14)
        plt.xlabel("Position X (mm)", fontsize=14)
        plt.legend()
        plt.savefig("Figures/insoles_position_activation_L.svg")
        plt.clf()

    activation_R = {
        "value": position_activation_R,
        "value_ref": position_activation_R_ref,
        "activation": activation_R,
        "name_activation": [
            "insole_R_2_up",
            "insole_R_3_up",
            "insole_R_4_up",
            "insole_R_5_up",
            "insole_R_6_up",
            "insole_R_7_mid",
            "insole_R_6_down",
            "insole_R_5_down",
            "insole_R_4_down",
            "insole_R_2_down",
        ],
    }

    activation_L = {
        "value": position_activation_L,
        "value_ref": position_activation_L_ref,
        "activation": activation_L,
        "name_activation": [
            "insole_L_1_mid",
            "insole_L_2_up",
            "insole_L_3_up",
            "insole_L_5_up",
            "insole_L_6_up",
            "insole_L_7_mid",
            "insole_L_6_down",
            "insole_L_5_down",
            "insole_L_4_down",
            "insole_L_3_down",
            "insole_L_2_down",
        ],
    }

    # Optimisation distance marker et sensor
    # distance_activation_sensor_R = []
    # distance_activation_sensor_L = []
    # for i in range(activation_R["value"].shape[1]):
    #     distance_activation_sensor_R.append(distance_activation_sensor(file_info_insole.iloc[6:8, :], activation_R["value"][:, i], activation_R["name_activation"][i]))
    #
    # for i in range(activation_L["value"].shape[1]):
    #     distance_activation_sensor_L.append(distance_activation_sensor(file_info_insole.iloc[4:6, :],
    #                                                               activation_L["value"][:, i], activation_L["name_activation"][i]))

    print("ok")

    return activation_R, activation_L  # , distance_activation_sensor_R, distance_activation_sensor_L
