import casadi as cas
from IPython import embed
import biorbd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from copy import copy
from scipy.optimize import fsolve
import math

# --- Functions --- #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

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
    index_post = []
    index_pre = []
    for valeur_proche in position_activation_y:
        differences = [abs(x - valeur_proche) for x in distance_sensor_y]
        index_proche_1 = int(min(range(len(differences)), key=differences.__getitem__))
        index_pre.append(index_proche_1)
        differences[index_proche_1] = float("inf")  # Exclure la première différence minimale
        index_proche_2 = int(min(range(len(differences)), key=differences.__getitem__))
        index_post.append(index_proche_2)

    pourcentage = []
    for i in range(len(index_post)):
        pourcentage.append(
            abs(distance_sensor_y[index_pre[i]] - position_activation_y[i])
            / abs(distance_sensor_y[index_pre[i]] - distance_sensor_y[index_post[i]])
        )

    resume = {
        "index_pre": index_pre,
        "index_post": index_post,
        "pourcentage": pourcentage,
    }

    return resume


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
        "distance_sensor_y": distance_sensor_y,
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
        "distance_sensor_y": distance_sensor_y,
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


def change_ref_marker(data):
    new_data = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        if i == 0:
            new_data[i] = data[i] - data[i]
        else:
            new_data[i] = data[i] - data[i - 1]
    return new_data


def distance_between_line_sensors(info_insole):
    distance_line_sensors_y = np.unique(info_insole.iloc[7, :], axis=0)
    distance_between_sensor = []
    for i in range(distance_line_sensors_y.shape[0]):
        distance_between_sensor.append(distance_line_sensors_y[i] - distance_line_sensors_y[0])
    return distance_between_sensor


def norm_2D(pointA, pointB) -> float:
    """
    Find the distance between two points
    :param pointA: A point with a coordinate on x and y
    :param pointB: A other point with a coordinate on x and y
    :return: The distance between this two points
    """
    norm = np.sqrt(((pointA[0] - pointB[0]) ** 2) + ((pointA[1] - pointB[1]) ** 2))
    return norm


def intersection_ellipse_line(line_points, ellipse_center, a, b, theta, FLAG_PLOT=False) -> list:
    """
    Find the intersection between a line and an ellipse
    :param line_points: Coordinate on x and y of two points and the line
    :param ellipse_center: Coordinate on x and y of the ellipse's center
    :param a: Major axis
    :param b: Minor axis
    :param theta: Angle of the ellipse
    :param FLAG_PLOT: If you want the plot of the intersection between a line and an ellipse
    :return: The coordinate on x and y of the intersection between a line and an ellipse
    """
    # Paramètres de la droite
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    # Calcule de la pente (m)
    m = (y2 - y1) / (x2 - x1)

    # Paramètres de l'ellipse
    h, k = ellipse_center

    # Redéfinir les points de la ligne pour passer par l'origine
    x1 -= h
    y1 -= k
    x2 -= h
    y2 -= k

    # Equations
    def equations(vars):
        x, y = vars

        eq1 = (
            (x * np.cos(theta) + y * np.sin(theta)) ** 2 / a**2
            + (y * np.cos(theta) - x * np.sin(theta)) ** 2 / b**2
            - 1
        )
        eq2 = y - y1 - m * (x - x1)
        return [eq1, eq2]

    # Utiliser fsolve de scipy pour trouver les solutions
    x, y = fsolve(equations, (x1, y1))
    intersection = x + h, y + k

    if FLAG_PLOT:
        ellipse = Ellipse(xy=(h, k), width=a, height=b, angle=theta * 180 / np.pi, facecolor="orange", alpha=0.5)

        func_droite = equation_droite((x1, y1), (x2, y2))
        # Intervalles de valeurs de x pour le traçage
        pos_x = abs(x2 - x1)
        x_values = np.linspace(-pos_x - 0.01, pos_x + 0.01, 100)

        # Évaluation de y pour chaque valeur de x
        y_values = [float(func_droite(x).full()[0]) for x in x_values]
        print(intersection)

        # new_ellipse = copy(ellipse_L)
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and markers")
        plt.plot(x_values, y_values, label="Droite")
        ax.scatter(h, k, color="red", label="centre ellipse")
        ax.scatter(x1, y1, color="orange", label="markers")
        ax.scatter(intersection[0], intersection[1], color="blue", label="intersection")
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")
        plt.legend()
        plt.grid(True)
        fig.clf()

    # Ramener les coordonnées des points d'intersection à l'original
    return intersection


def equation_droite(pointA, pointB):
    """
    Equation of a line
    :param pointA: Coordinate on x and y of a point on the line
    :param pointB: Coordinate on x and y of an other point on the line
    :return: A function casadi of the equation of a line
    """
    xA, yA = pointA
    xB, yB = pointB

    # Variables symboliques
    x = cas.MX.sym("x")

    # Calcule de la pente (m)
    m = (yB - yA) / (xB - xA)

    # Calcule de l'ordonnée
    b = yA - m * xA

    # Equation de la droite
    y = m * x + b

    # Création fonction casadi
    func = cas.Function("f", [x], [y])
    return func


def find_index_by_name(list: list, word: str):
    """
    Find the index of an element in a list from its name
    :param list: List where is the word
    :param word: Word
    :return: The index of the word into a list
    """
    index_finding = [index for index, markers in enumerate(list) if word in markers]
    return index_finding


def points_to_ellipse(data, fig_name, markers_name, FLAG_PLOT: False) -> list:
    """

    :param data: File with the data
    :param fig_name: The name you want for the figure
    :param markers_name:
    :param FLAG_PLOT: If you want to save the figure
    :return: The parameters of the ellipse (a, b, center ellipse, theta)
    """

    data = np.array(data.T)
    norm = []
    position = ["up", "down", "mid"]
    index_doublon = ["2", "3", "4", "5", "6"]
    index_marker_parameter = {}
    for i in range(len(position)):
        index_marker_parameter["marker_" + str(position[i])] = find_index_by_name(markers_name, position[i])
    for i in range(len(index_doublon)):
        index_marker_parameter["marker_" + str(index_doublon[i])] = find_index_by_name(markers_name, index_doublon[i])
        norm.append(
            norm_2D(
                data[index_marker_parameter["marker_" + str(index_doublon[i])][0], :],
                data[index_marker_parameter["marker_" + str(index_doublon[i])][1], :],
            )
        )
    print(norm)

    # Generate an initial guess for the ellipse parameters
    theta_gauss = 0
    width_gauss = 1
    height_gauss = 1

    # State the optimization problem with the following variables
    # Angle (theta)
    # width of the ellipse (a)
    # height of the ellipse (b)
    # x center of the ellipse (xc)
    # y center of the ellipse (yc)
    ellipse_param = cas.MX.sym("parameters", 5)

    # centers_index = [
    #     [index_marker_parameter["marker_up"] + index_marker_parameter["marker_mid"], "up+mid"],
    #     [index_marker_parameter["marker_down"] + index_marker_parameter["marker_mid"], "down+mid"],
    #     [index_marker_parameter["marker_up"] + index_marker_parameter["marker_mid"] + index_marker_parameter[
    #          "marker_up"], "all"]]

    centers_index = [
        [index_marker_parameter["marker_up"], "up"],
        [index_marker_parameter["marker_down"], "down"],
        [index_marker_parameter["marker_up"] + index_marker_parameter["marker_down"], "up_down"],
    ]

    ellipse = []

    for i in range(len(centers_index)):
        # centers = data[markers_penalized[i], :]
        centers = data[centers_index[i][0], :]
        mean_centers = np.mean(centers, axis=0)
        x0 = np.array([theta_gauss, width_gauss, height_gauss, mean_centers[0], mean_centers[1]])

        # Objective (minimize squared distance between points and ellipse boundary)
        f = 0
        for indices_this_time in range(centers.shape[0]):
            cos_angle = cas.cos(np.pi - ellipse_param[0])
            sin_angle = cas.sin(np.pi - ellipse_param[0])

            xc = centers[indices_this_time, 0] - ellipse_param[3]
            yc = centers[indices_this_time, 1] - ellipse_param[4]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            f += (
                (xct**2 / (((ellipse_param[1]) / 2) ** 2))
                + (yct**2 / (((ellipse_param[2]) / 2) ** 2))
                - cas.sqrt(xc**2 + yc**2)
            ) ** 2

        nlp = {"x": ellipse_param, "f": f}
        opts = {"ipopt.print_level": 5}
        solver = cas.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(
            x0=x0,
            lbx=[-np.pi, 0, 0, mean_centers[0] - 0.5, mean_centers[1] - 0.5],
            ubx=[np.pi, 0.2, 0.3, mean_centers[0] + 0.5, mean_centers[1] + 0.5],
        )

        if solver.stats()["success"]:
            success_out = True
            theta_opt = float(sol["x"][0])
            width_opt = float(sol["x"][1])
            height_opt = float(sol["x"][2])
            center_x_opt = float(sol["x"][3])
            center_y_opt = float(sol["x"][4])
        else:
            print("Ellipse did not converge, trying again")

        parameters_ellipse = {
            "a": width_opt,
            "b": height_opt,
            "center_x_ellipse": center_x_opt,
            "center_y_ellipse": center_y_opt,
            "angle": theta_opt,
            "index_markers": centers_index[i][0],
            "type_markers": centers_index[i][1],
        }
        print("Paramètres optimaux de l'ellipse: \t" + fig_name)
        print("Type ellipse: " + str(centers_index[i][1]))
        print("Grande diagonale de l'ellipse: " + str(width_opt))
        print("Petite diagonale de l'ellipse: " + str(height_opt))
        print("Centre x de l'ellipse: " + str(center_x_opt))
        print("Centre y de l'ellipse: " + str(center_y_opt))
        print("Angle theta de l'ellipse: " + str(theta_opt))
        ellipse.append(parameters_ellipse)

    # Plotting the ellipse
    if FLAG_PLOT:
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and markers")

        # Creation de l'ellipse
        ellipse_up = Ellipse(
            xy=(ellipse[0]["center_x_ellipse"], ellipse[0]["center_y_ellipse"]),
            width=ellipse[0]["a"] / 2,
            height=ellipse[0]["b"] / 2,
            angle=ellipse[0]["angle"] * 180 / np.pi,
            facecolor="red",
            alpha=0.5,
        )
        ellipse_down = Ellipse(
            xy=(ellipse[1]["center_x_ellipse"], ellipse[1]["center_y_ellipse"]),
            width=ellipse[1]["a"] / 2,
            height=ellipse[1]["b"] / 2,
            angle=ellipse[1]["angle"] * 180 / np.pi,
            facecolor="blue",
            alpha=0.5,
        )
        ellipse_all = Ellipse(
            xy=(ellipse[2]["center_x_ellipse"], ellipse[2]["center_y_ellipse"]),
            width=ellipse[2]["a"] / 2,
            height=ellipse[2]["b"] / 2,
            angle=ellipse[2]["angle"] * 180 / np.pi,
            facecolor="orange",
            alpha=0.5,
        )

        # Integration markers
        up_markers = ax.plot(
            data[index_marker_parameter["marker_up"], 0],
            data[index_marker_parameter["marker_up"], 1],
            "ro",
            label="markers up",
        )
        down_markers = ax.plot(
            data[index_marker_parameter["marker_down"], 0],
            data[index_marker_parameter["marker_down"], 1],
            "bo",
            label="markers down",
        )
        mid_markers = ax.plot(
            data[index_marker_parameter["marker_mid"], 0],
            data[index_marker_parameter["marker_mid"], 1],
            "go",
            label="markers mid",
        )

        # Ajout de l'ellipse aux axes
        ax.add_patch(ellipse_up)
        ax.add_patch(ellipse_down)
        ax.add_patch(ellipse_all)
        ax.set_aspect("equal")

        # Définition des parametres des axes
        # ax.set_xlim((center_x_opt - width_opt) - 0.1, (center_x_opt + width_opt) + 0.1)
        # ax.set_ylim((center_y_opt - height_opt) - 0.1, (center_y_opt + height_opt) + 0.1)
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")
        plt.legend(handles=[up_markers[0], down_markers[0], mid_markers[0]])

        # Affichage de la figure
        plt.savefig(fig_name + ".svg")
        # plt.show()

    return ellipse


def minimize_distance(
    position_markers,
    position_activation,
    coordonnees_insole_45_y,
    ellipse_center,
    ellipse_axes,
    ellipse_angle,
    FLAG_PLOT=False,
):
    """

    Parameters
    ----------
    position_markers_y: position markers in y, nedd to be referenced to the first activation of the insole
    ellipse_center: Value of the center of the ellipse
    ellipse_axes: Value of the axes of the ellipse (a and b)
    ellipse_angle: Angle of the ellipse

    Returns: angle optimized to put the activation on the insole
    -------

    """
    # TODO: change insole ref to put the zero on the first line
    x, y = change_ref_marker(position_markers[0]), change_ref_marker(position_markers[1])

    # TODO: compute the "Y position"on the sensors on the insole
    # insole_sensors = np.linspace(0, 0.3, 18)  # To be changed
    insole_sensors = coordonnees_insole_45_y

    # Optimization variables
    angle_to_put_zero = cas.MX.sym("angle_to_put_zero", 1)
    x_sensors = cas.MX.sym("x_sensors", 18)  # nly for implicit constraint
    y_sensors = cas.MX.sym("y_sensors", 18 - 1)  # nly for implicit constraint

    f = 0
    g = []
    lbg = []
    ubg = []

    cos_angle = cas.cos(np.pi - ellipse_angle)
    sin_angle = cas.sin(np.pi - ellipse_angle)

    xc_0 = x_sensors[0] - ellipse_center[0]
    yc_0 = x_sensors[0] * cas.tan(angle_to_put_zero) - ellipse_center[1]

    xct = xc_0 * cos_angle - yc_0 * sin_angle
    yct = xc_0 * sin_angle + yc_0 * cos_angle

    g += [
        (
            (xct**2 / (((ellipse_axes[0]) / 2) ** 2))
            + (yct**2 / (((ellipse_axes[1]) / 2) ** 2))
            - cas.sqrt(xc_0**2 + yc_0**2)
        )
        ** 2
    ]
    lbg += [1]
    ubg += [1]

    # Objective (minimize distance between two points)
    for i_sensor in range(1, 18):

        cos_angle = cas.cos(np.pi - ellipse_angle)
        sin_angle = cas.sin(np.pi - ellipse_angle)

        xc = x_sensors[i_sensor] - ellipse_center[0]
        yc = y_sensors[i_sensor - 1] - ellipse_center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        g += [
            (
                (xct**2 / (((ellipse_axes[0]) / 2) ** 2))
                + (yct**2 / (((ellipse_axes[1]) / 2) ** 2))
                - cas.sqrt(xc**2 + yc**2)
            )
            ** 2
        ]
        lbg += [1]
        ubg += [1]

        # TODO: make sure it goes in the right direction, regarde si on prend le bonne indice
        tata = cas.MX.zeros(2)
        tata[0] = xc - xc_0
        tata[1] = yc - yc_0
        g += [cas.norm_2(insole_sensors[i_sensor] - insole_sensors[i_sensor - 1]) - (cas.norm_2(tata))]
        lbg += [0]
        ubg += [0]

        xc_0 = xc
        yc_0 = yc

    # TODO change for real values + loop
    #  disons que la premiere activation se trouve exactement entre le sensor 3 et 4
    markers = 0
    activation_position = cas.MX.zeros(2)
    for i in range(len(position_activation["index_pre"])):
        activation_position[0] = x_sensors[position_activation["index_pre"][i]] + position_activation["pourcentage"][
            i
        ] * (x_sensors[position_activation["index_post"][i]] - x_sensors[position_activation["index_pre"][i]])
        activation_position[1] = y_sensors[position_activation["index_pre"][i]] + position_activation["pourcentage"][
            i
        ] * (y_sensors[position_activation["index_post"][i]] - y_sensors[position_activation["index_pre"][i]])
        distance_marker_sensor = cas.sqrt(
            (x[markers] - activation_position[0]) ** 2 + (y[markers] - activation_position[1]) ** 2
        )
        f += distance_marker_sensor

    nlp = {"x": cas.vertcat(angle_to_put_zero, x_sensors, y_sensors), "f": f, "g": cas.vertcat(*g)}
    opts = {"ipopt.print_level": 5}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)
    x0 = np.zeros((18 * 2,))  # Initial guess for the optimization variables

    sol = solver(x0=x0, lbx=[-np.inf] * 18 * 2, ubx=[np.inf] * 18 * 2, lbg=lbg, ubg=ubg)
    # if FLAG_PLOT: # Dessiner l'ellipse,  # Mettre les markers, # Mettre les points d'activation, # Mettre les activations estimées

    if solver.stats()["success"]:
        output_variables = float(sol["x"])
        print("output_variables", output_variables)
        return output_variables
    else:
        print("Optimization did not converge")
        return None


def find_tangent(ellipse_center, ellipse_axes, ellipse_angle, point, fig_name: str, FLAG_PLOT=False):
    """
    Find the tangent of a point on the edge of a ellipse
    :param ellipse_center: Coordinate on x and y of the ellipse's center
    :param ellipse_axes: Values of the major and the minor axis
    :param ellipse_angle: Angle of the ellipse
    :param point: Coordinate on x and y of a point on the edge of the ellipse
    :param FLAG_PLOT: If you want the plot of the tagent of a point of the edge of the ellipse
    :return: The slope of the tangent
    """
    cx, cy = ellipse_center
    a, b = ellipse_axes
    x, y = point

    # Rotation inverse et translation inverse pour obtenir les coordonnées du point dans le système de l'ellipse
    x1 = (x - cx) * np.cos(ellipse_angle) + (y - cy) * np.sin(ellipse_angle)
    y1 = -(x - cx) * np.sin(ellipse_angle) + (y - cy) * np.cos(ellipse_angle)

    # Coefficients pour l'équation de la tangente
    A = x1 / a**2
    B = y1 / b**2

    # Slope of the tangent line in the ellipse's coordinate system
    slope_ellipse = -A / B

    # Convert the slope back to the original coordinate system
    slope = (slope_ellipse * np.cos(ellipse_angle) - np.sin(ellipse_angle)) / (
        np.cos(ellipse_angle) + slope_ellipse * np.sin(ellipse_angle)
    )

    if FLAG_PLOT:
        # Generate points for the tangent line
        t = np.linspace(x - 0.1, x + 0.1, 100)
        tangent_line = y + slope * (t - x)

        # Tracer l'ellipse et la tangente
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and the tangente of a one marker")

        # Tracer l'ellipse
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_angle * 180 / np.pi, facecolor="red", alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Tracer le point d'intersection
        ax.scatter(x, y, color="blue", label="intersection")

        # Tracer la tangente
        ax.plot(t, tangent_line, color="blue", label="tangente")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        # plt.show()

    return slope_ellipse, slope


def find_perpendiculaire_to_tangente(
    tangent_slope, point, ellipse_axes, ellipse_angle, ellipse_center, fig_name: str, FLAG_PLOT=False
):
    """

    :param tangent_slope: The slope of the tangent
    :param point: Coordinate on x and y of a point on the edge of the ellipse
    :param ellipse_axes: Values of the major and the minor axis
    :param ellipse_angle: Values of the major and the minor axis
    :param ellipse_center: Coordinate on x and y of the ellipse's center
    :param FLAG_PLOT: If you want the plot of the tagent and the perpendicular of a point of the edge of the ellipse
    :return:
    """
    x, y = point
    cx, cy = ellipse_center
    a, b = ellipse_axes

    # Pente de la perpendiculaire à la tangente
    perpendicular_slope = -1 / tangent_slope

    # Génération des points pour la perpendiculaire à la tangente
    t = np.linspace(x - 0.1, x + 0.1, 100)
    perpendicular_line = y + perpendicular_slope * (t - x)
    tangent_line = y + tangent_slope * (t - x)

    if FLAG_PLOT:
        # Tracer l'ellipse, la tangente et la perpendiculaire à la tangente
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole, tangente and perpendiculaire")

        # Tracer l'ellipse
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_angle * 180 / np.pi, facecolor="red", alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Tracer le point d'intersection
        ax.scatter(x, y, color="blue", label="intersection")

        # Tracer la tangente
        ax.plot(t, tangent_line, color="blue", label="tagente")

        # Tracer la perpendiculaire à la tangente
        ax.plot(t, perpendicular_line, color="orange", label="perpendiculaire")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        # plt.show()

    return perpendicular_line


def position_insole(marker_list: list, model):
    """

    :param marker_list: List marker insole with only index of markers in the model
    :param model: Biorbd model with the markers
    :return: The position of the insole's center and the position of all markers of the insole
    """
    if all(isinstance(valeur, int) for valeur in marker_list):
        position_markers = np.zeros(shape=(3, len(marker_list)))
        for i in range(len(marker_list)):
            position_markers[:, i] = model.markers()[marker_list[i]].to_array()
        center = np.mean(position_markers, axis=1)
    else:
        position_markers = None
        center = None
        print("ERROR: La liste doit contenir les index des markers dans le modèle !")
    return center, position_markers
