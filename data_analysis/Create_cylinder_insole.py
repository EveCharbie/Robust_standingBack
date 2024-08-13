import casadi as cas
from matplotlib.patches import Ellipse
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# --- Functions --- #
def lissage(signal_brut, L):
    """
    Smooth the signal
    Parameters
    ----------
    signal_brut:
        Signal to smooth
    L:
        Number of values to take into account

    Returns
    -------
    res:
        Smoothed signal
    """
    res = np.copy(signal_brut)  # duplication des valeurs
    for i in range(1, len(signal_brut) - 1):  # toutes les valeurs sauf la première et la dernière
        L_g = min(i, L)  # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut) - i - 1, L)  # nombre de valeurs disponibles à droite
        Li = min(L_g, L_d)
        res[i] = np.sum(signal_brut[i - Li : i + Li + 1]) / (2 * Li + 1)
    return res


def distance_activation_sensor(coordonnes_insoles, position_activation, name_activation):
    """
    Calculate the distance between the activation and the sensors
    Parameters
    ----------
    coordonnes_insoles:
        Coordonnes of the sensors
    position_activation:
        Position of the activation
    name_activation:
        Name of the activation

    Returns
    -------
    distance:
        Distance between the activation and the sensors
    """
    distance = {
        "value": abs(position_activation[:, np.newaxis] - coordonnes_insoles),
        "activation_name": name_activation,
    }
    return distance


def find_activation_sensor(distance_sensor_y, position_activation_y):
    """
    Find the position of the activation in relation to a selected sensor
    Parameters
    ----------
    distance_sensor_y:
        Distance between the activation and the sensors
    position_activation_y:
        Position of the activation

    Returns
    -------
    resume:
        Resume of the sensor to the activation
    """
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


def cartography_insole(file_insole, file_info_insole, FLAG_PLOT=False):
    """
    Plot the cartography of the insoles
    Parameters
    ----------
    file_insole:
        File of the insole
    file_info_insole:
        File of the information of the insole
    FLAG_PLOT:
        Flag to plot the results

    Returns
    -------

    """

    sensor_45 = file_insole.columns[
        1:-1
    ].to_list()  #  List sensor on the insole size 45 (without columns Sync and Time)
    coordonnees_insole = file_info_insole.loc[0:7, sensor_45]

    # Plot : Insoles
    if FLAG_PLOT:
        fig, axs = plt.subplots(2)
        fig.suptitle("Insoles cartography")

        # Subplot 1 : insoles R
        axs[0].plot(coordonnees_insole.iloc[7, :], coordonnees_insole.iloc[6, :], "ro")
        axs[0].set_ylabel("Position Y (mm)", fontsize=14)
        axs[0].title.set_text("Insole Right")

        # Subplot 2 : insoles L
        axs[1].plot(coordonnees_insole.iloc[5, :], coordonnees_insole.iloc[4, :], "ro")
        axs[1].set_xlabel("Position X (mm)", fontsize=14)
        axs[1].set_ylabel("Position Y (mm)", fontsize=14)
        axs[1].title.set_text("Insole Left")
        plt.savefig("Figures/cartography.svg")
        fig.clf()
    return


def position_activation(file_insole_R, file_insole_L, file_info_insole, FLAG_PLOT=False):
    """
    Find the position of the activation
    Parameters
    ----------
    file_insole_R:
        File of the right insole
    file_insole_L:
        File of the left insole
    file_info_insole:
        File of the information of the insole
    FLAG_PLOT:
        Flag to plot the results

    Returns
    -------
    position_activation:
        Position of the activation
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
        plt.close(fig)

    position_activation_R = np.zeros(shape=(2, peaks_R.shape[0]))
    position_activation_L = np.zeros(shape=(2, peaks_L.shape[0]))
    position_activation_R_ref = np.zeros(shape=(2, peaks_R.shape[0]))
    position_activation_L_ref = np.zeros(shape=(2, peaks_L.shape[0]))

    for i in range(peaks_R.shape[0]):
        # Take 10 frames before and after the peak and find sensors activated
        insole_R_activate_peak = markers_insole_R_baseline.loc[:, filtered_sensor_min_max_R.columns.to_list()][
            peaks_R[i] - 10 : peaks_R[i] + 10
        ]
        # Select columns with values greater than zero
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

    # remove column 5 (sensor located at end of base plate)
    position_activation_R_ref = np.delete(position_activation_R_ref, 5, axis=1)
    position_activation_L_ref = np.delete(position_activation_L_ref, 5, axis=1)

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
        plt.close(fig)

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
            "insole_R_up_2",
            "insole_R_up_3",
            "insole_R_up_4",
            "insole_R_up_5",
            "insole_R_up_6",
            "insole_R_mid_7",
            "insole_R_down_6",
            "insole_R_down_5",
            "insole_R_down_4",
            "insole_R_down_2",
        ],
        "position_activation": position_activation_R / 1000,
        "all_sensors_positions": np.array(file_info_insole.iloc[7, :], file_info_insole.iloc[6, :]) / 1000,
    }

    activation_L = {
        "value": position_activation_L,
        "value_ref": position_activation_L_ref,
        "activation": activation_L,
        "distance_sensor_y": distance_sensor_y,
        "name_activation": [
            "insole_G_mid_1",
            "insole_G_up_2",
            "insole_G_up_3",
            "insole_G_up_5",
            "insole_G_up_6",
            "insole_G_mid_7",
            "insole_G_down_6",
            "insole_G_down_5",
            "insole_G_down_4",
            "insole_G_down_3",
            "insole_G_down_2",
        ],
        "position_activation": position_activation_L / 1000,
        "all_sensors_positions": np.array(file_info_insole.iloc[5, :], file_info_insole.iloc[4, :]) / 1000,
    }

    return activation_R, activation_L


def change_ref_marker(data):
    """
    Change the reference of the marker to the first value of the marker
    Parameters
    ----------
    data:
        Data of the markers

    Returns
    -------
    new_data:
        Data of the markers with the reference changed
    """
    new_data = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        if i == 0:
            new_data[i] = data[i] - data[i]
        else:
            new_data[i] = data[i] - data[i - 1]
    return new_data


def distance_between_line_sensors(info_insole):
    """
    Find the distance between the line sensors
    Parameters
    ----------
    info_insole:
        Information of the insole

    Returns
    -------
    distance_between_sensor:
        Distance between the line sensors
    """
    distance_line_sensors_y = np.unique(info_insole.iloc[7, :], axis=0)
    distance_between_sensor = []
    for i in range(distance_line_sensors_y.shape[0]):
        distance_between_sensor.append(distance_line_sensors_y[i] - distance_line_sensors_y[0])
    return distance_between_sensor


def norm_2D(pointA, pointB) -> float:
    """
    Find the norm between two points
    Parameters
    ----------
    pointA:
        Coordinate on x and y of the point A
    pointB:
        Coordinate on x and y of the point B

    Returns
    -------
    norm:
        Norm between the two points
    """
    norm = np.sqrt(((pointA[0] - pointB[0]) ** 2) + ((pointA[1] - pointB[1]) ** 2))
    return norm


def intersection_ellipse_line(line_points, ellipse_center, a, b, theta, FLAG_PLOT=False) -> list:
    """
    Find the intersection between a line and an ellipse
    Parameters
    ----------
    line_points:
        Points of the line
    ellipse_center:
        Center of the ellipse
    a:
        Semi-major axis of the ellipse
    b:
        Semi-minor axis of the ellipse
    theta:
        Angle of the ellipse
    FLAG_PLOT:
        Flag to plot the intersection

    Returns
    -------
    intersection_points:
        Points of the intersection
    """
    # Parameters line
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    # Calcule de la pente (m)
    m = (y2 - y1) / (x2 - x1)

    # Slope calculation (m)
    h, k = ellipse_center

    # Redefine line points to pass through origin
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

    # Use scipy's fsolve to find solutions
    x, y = fsolve(equations, (x1, y1))
    intersection = x + h, y + k

    # Plotting
    if FLAG_PLOT:
        ellipse = Ellipse(xy=(h, k), width=a, height=b, angle=theta * 180 / np.pi, facecolor="orange", alpha=0.5)

        func_droite = equation_droite((x1, y1), (x2, y2))
        # Ranges of x values for tracing
        pos_x = abs(x2 - x1)
        x_values = np.linspace(-pos_x - 0.01, pos_x + 0.01, 100)

        # Evaluation of y for each value of x
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
        plt.close(fig)

    return intersection


def equation_droite(pointA, pointB):
    """
    Find the equation of a line from two points with a casadi function
    Parameters
    ----------
    pointA:
        Coordinate on x and y of the point A
    pointB:
        Coordinate on x and y of the point B

    Returns
    -------
    func:
        Equation of the line
    """

    xA, yA = pointA
    xB, yB = pointB

    # Symbolic variables
    x = cas.MX.sym("x")

    # Slope calculation (m)
    m = (yB - yA) / (xB - xA)

    # Calculation of ordinate
    b = yA - m * xA

    # Equation of the line
    y = m * x + b

    # Create casadi function
    func = cas.Function("f", [x], [y])

    return func


def find_index_by_name(list: list, word: str):
    """
    Find the index of a word in a list
    Parameters
    ----------
    list:
        List of words
    word:
        Word to find

    Returns
    -------
    index_finding:
        Index of the word in the list
    """
    index_finding = [index for index, markers in enumerate(list) if word in markers]
    return index_finding


def points_to_ellipse(markers_insole_xy, fig_name, markers_name, FLAG_PLOT: False) -> list:
    """
    Find the ellipse parameters from the markers
    Parameters
    ----------
    markers_insole_xy:
        Data of the markers
    fig_name:
        Name of the figure
    markers_name:
        Name of the markers
    FLAG_PLOT:
        Flag to plot the ellipse

    Returns
    -------
    ellipse_parameters:
        Parameters of the ellipse

    ellipse equation:
        (((x-xc)cos(theta) + (y-yc)sin(theta))**2) / a**2 + (((x-xc)sin(theta) - (y-yc)cos(theta))**2) / b**2 = 1

    """

    markers_insole_xy = np.array(markers_insole_xy.T)
    norm = []
    position = ["up", "down", "mid"]
    up_down_paired_markers_index = ["2", "3", "4", "5", "6"]
    index_marker_parameter = {}
    for i in range(len(position)):
        index_marker_parameter["marker_" + str(position[i])] = find_index_by_name(markers_name, position[i])
    for i in range(len(up_down_paired_markers_index)):
        index_marker_parameter["marker_" + str(up_down_paired_markers_index[i])] = find_index_by_name(markers_name, up_down_paired_markers_index[i])
        norm.append(
            norm_2D(
                markers_insole_xy[index_marker_parameter["marker_" + str(up_down_paired_markers_index[i])][0], :],
                markers_insole_xy[index_marker_parameter["marker_" + str(up_down_paired_markers_index[i])][1], :],
            )
        )
    print(f"The horizontal distance between the up-down marker pairs are : {np.array(norm) * 100} cm")
    print("Please note that the markers were not perfectly aligned one on top of the other, so this measure is not indicative of the tilt of the tibia.")

    # State the optimization problem with the following variables
    # Angle (theta)
    # width of the ellipse (a)
    # height of the ellipse (b)
    # x center of the ellipse (xc)
    # y center of the ellipse (yc)
    ellipse_param = cas.MX.sym("parameters", 5)

    ellipses_markers_index = [
        [index_marker_parameter["marker_up"] + index_marker_parameter["marker_mid"], "up"],
        [index_marker_parameter["marker_down"] + index_marker_parameter["marker_mid"], "down"],
        [index_marker_parameter["marker_up"] + index_marker_parameter["marker_down"] + index_marker_parameter["marker_mid"], "up_down"],
    ]

    ellipse = []

    for i in range(len(ellipses_markers_index)):
        markers_for_this_ellispe = markers_insole_xy[ellipses_markers_index[i][0], :]
        mean_marker_position = np.mean(markers_for_this_ellispe, axis=0)

        # Generate a good intial guess for the ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(np.cov(markers_for_this_ellispe[:, 0], markers_for_this_ellispe[:, 1]))
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta_gauss = np.arctan2(vy, vx)
        width_gauss, height_gauss = 2 * 2 * np.sqrt(eigvals)
        x0 = np.array([theta_gauss, width_gauss, height_gauss, mean_marker_position[0], mean_marker_position[1]])

        # Min area version
        # Objective (minimize area of the ellipse)
        f = np.pi * ellipse_param[1] * ellipse_param[2]  # ellipse air = pi * a * b

        # Constraints (all point are inside the ellipse)
        g = []
        lbg = []
        ubg = []
        for indices_this_time in range(markers_for_this_ellispe.shape[0]):

            cos_angle = cas.cos(np.pi-ellipse_param[0])
            sin_angle = cas.sin(np.pi-ellipse_param[0])

            xc = markers_for_this_ellispe[indices_this_time, 0] - ellipse_param[3]
            yc = markers_for_this_ellispe[indices_this_time, 1] - ellipse_param[4]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            rad_cc = (xct ** 2 / (ellipse_param[1] / 2.) ** 2) + (yct ** 2 / (ellipse_param[2] / 2.) ** 2)

            g += [rad_cc]

            lbg += [-cas.inf]
            ubg += [1]

        nlp = {"x": ellipse_param, "f": f, "g": cas.vertcat(*g)}
        opts = {"ipopt.print_level": 5}
        solver = cas.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=x0, lbx=[-2 * np.pi, 0, 0, mean_marker_position[0]-0.01, mean_marker_position[1]-0.01], ubx=[np.pi, 2, 2, mean_marker_position[0]+0.01,mean_marker_position[0]+0.01], lbg=lbg, ubg=ubg)

        theta_opt_first_pass = float(sol["x"][0])
        width_opt_first_pass = float(sol["x"][1])
        height_opt_first_pass = float(sol["x"][2])
        center_x_opt_first_pass = float(sol["x"][3])
        center_y_opt_first_pass = float(sol["x"][4])

        x0 = np.array([theta_opt_first_pass, width_opt_first_pass, height_opt_first_pass, center_x_opt_first_pass, center_y_opt_first_pass])
        theta_range = np.sort(np.array([theta_opt_first_pass * 0.8, theta_opt_first_pass * 1.2]))
        width_range = np.sort(np.array([width_opt_first_pass * 0.8, width_opt_first_pass * 1.2]))
        height_range = np.sort(np.array([height_opt_first_pass * 0.8, height_opt_first_pass * 1.2]))
        center_x_range = np.sort(np.array([center_x_opt_first_pass * 0.8, center_x_opt_first_pass * 1.2]))
        center_y_range = np.sort(np.array([center_y_opt_first_pass * 0.8, center_y_opt_first_pass * 1.2]))
        lbx = np.array([theta_range[0], width_range[0], height_range[0], center_x_range[0], center_y_range[0]])
        ubx = np.array([theta_range[1], width_range[1], height_range[1], center_x_range[1], center_y_range[1]])

        # new version
        f = 0
        for indices_this_time in range(markers_for_this_ellispe.shape[0]):

            cos_angle = cas.cos(np.pi-ellipse_param[0])
            sin_angle = cas.sin(np.pi-ellipse_param[0])

            # Position of the marker in the referential of the ellipse center
            xc = markers_for_this_ellispe[indices_this_time, 0] - ellipse_param[3]
            yc = markers_for_this_ellispe[indices_this_time, 1] - ellipse_param[4]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            rad_cc = (xct ** 2 / (ellipse_param[1] / 2.) ** 2) + (yct ** 2 / (ellipse_param[2] / 2.) ** 2)

            # Sum of squared distances
            f += (rad_cc - 1) ** 2

        nlp = {"x": ellipse_param, "f": f}
        opts = {"ipopt.print_level": 5}
        solver = cas.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        theta_opt = float(sol["x"][0])
        width_opt = float(sol["x"][1])
        height_opt = float(sol["x"][2])
        center_x_opt = float(sol["x"][3])
        center_y_opt = float(sol["x"][4])
        if not solver.stats()["success"]:
            raise RuntimeError("Ellipse did not converge, trying again !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        parameters_ellipse = {
            "a": width_opt,
            "b": height_opt,
            "center_x_ellipse": center_x_opt,
            "center_y_ellipse": center_y_opt,
            "angle": theta_opt,
            "index_markers": ellipses_markers_index[i][0],
            "type_markers": ellipses_markers_index[i][1],
        }
        print("Paramètres optimaux de l'ellipse: \t" + fig_name)
        print("Type ellipse: " + str(ellipses_markers_index[i][1]))
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
            width=ellipse[0]["a"],
            height=ellipse[0]["b"],
            angle=(np.pi - ellipse[0]["angle"]) * 180 / np.pi,
            facecolor="red",
            alpha=0.5,
        )
        ellipse_down = Ellipse(
            xy=(ellipse[1]["center_x_ellipse"], ellipse[1]["center_y_ellipse"]),
            width=ellipse[1]["a"],
            height=ellipse[1]["b"],
            angle=(np.pi - ellipse[1]["angle"]) * 180 / np.pi,
            facecolor="blue",
            alpha=0.5,
        )
        ellipse_all = Ellipse(
            xy=(ellipse[2]["center_x_ellipse"], ellipse[2]["center_y_ellipse"]),
            width=ellipse[2]["a"],
            height=ellipse[2]["b"],
            angle=(np.pi - ellipse[2]["angle"]) * 180 / np.pi,
            facecolor="green",
            alpha=0.5,
        )

        # Integration markers
        up_markers = ax.plot(
            markers_insole_xy[index_marker_parameter["marker_up"], 0],
            markers_insole_xy[index_marker_parameter["marker_up"], 1],
            "ro",
            label="markers up",
        )
        down_markers = ax.plot(
            markers_insole_xy[index_marker_parameter["marker_down"], 0],
            markers_insole_xy[index_marker_parameter["marker_down"], 1],
            "bo",
            label="markers down",
        )
        mid_markers = ax.plot(
            markers_insole_xy[index_marker_parameter["marker_mid"], 0],
            markers_insole_xy[index_marker_parameter["marker_mid"], 1],
            "go",
            label="markers mid",
        )

        # Add ellipse to axes
        ax.add_patch(ellipse_up)
        ax.add_patch(ellipse_down)
        ax.add_patch(ellipse_all)
        ax.set_aspect("equal")

        # Defining axis parameters
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")
        plt.legend(handles=[up_markers[0], down_markers[0], mid_markers[0]])

        # Displaying and saving the figure
        plt.savefig("Figures/" + fig_name + ".png")
        plt.close()

    return ellipse


def minimize_distance(
    position_markers,
    markers_insole_name,
    insole_activations,
    ellipse_theta,
    ellipse_width,
    ellipse_height,
    ellipse_center_x,
    ellipse_center_y,
    fig_name,
    FLAG_PLOT=True,
):
    """
    Roll the insoles around the tibia ellipse.
    The position of the MoCap markers should be as close as possible to the insoles activation position as they were actually superimposed.
    Parameters
    ----------
    TODO: doc
    FLAG_PLOT:
        Flag to plot the results

    Returns
    -------

    """
    # Sensor to marker correspondence
    sensor_columns = np.sort(np.array(list(set(list(insole_activations["all_sensors_positions"])))))  # Get single values of column position (x coordinate) in meters
    names_in_order = ["mid_1", "up_2", "down_2", "up_3", "down_3", "up_4", "down_4", "up_5", "down_5", "up_6", "down_6", "mid_7"]

    sensors_and_markers_index = {}
    nb_sensors = 0
    for name in names_in_order:
        sensor_idx = None
        for i, sensor in enumerate(insole_activations["name_activation"]):
            if name in sensor:
                sensor_idx = i
                nb_sensors += 1
        marker_idx = None
        for i, marker in enumerate(markers_insole_name):
            if name in marker:
                marker_idx = i
        sensors_and_markers_index[name] = [sensor_idx, marker_idx]

    # Optimization variables
    x_sensors = cas.MX.sym("x_sensors", nb_sensors)
    y_sensors = cas.MX.sym("y_sensors", nb_sensors)
    x_columns = cas.MX.sym("x_columns", len(sensor_columns))
    y_columns = cas.MX.sym("y_columns", len(sensor_columns))

    f = 0
    g = []
    lbg = []
    ubg = []
    x0 = np.zeros((nb_sensors * 2 + len(sensor_columns) * 2, ))

    cos_angle = cas.cos(np.pi - ellipse_theta)
    sin_angle = cas.sin(np.pi - ellipse_theta)

    # Objective (minimize distance between two points)
    for name in names_in_order:

        i_sensor = sensors_and_markers_index[name][0]
        if i_sensor is None:
            continue

        # Minimize distance between markers and insoles activations
        if sensors_and_markers_index[name][1] is not None:
            i_marker = sensors_and_markers_index[name][1]
            f += (x_sensors[i_sensor] - position_markers[0, i_marker]) ** 2 + (y_sensors[i_sensor] - position_markers[1, i_marker]) ** 2
            x0[i_sensor] = position_markers[0, i_marker]
            x0[i_sensor + nb_sensors] = position_markers[1, i_marker]

        # Impose that the distance between the sensors and the columns are the same as those on the insoles
        if np.round(insole_activations["position_activation"][1, i_sensor], 7) in np.round(sensor_columns, 7):
            i_column_equal = np.where(np.round(insole_activations["position_activation"][1, i_sensor], 7) == np.round(sensor_columns, 7))[0][0]
            g += [x_sensors[i_sensor] - x_columns[i_column_equal], y_sensors[i_sensor] - y_columns[i_column_equal]]
            lbg += [0, 0]
            ubg += [0, 0]
        else:
            i_sensor_before = np.where(insole_activations["position_activation"][1, i_sensor] > sensor_columns)[0][-1]
            distance_sensors_before = (x_sensors[i_sensor] - x_columns[i_sensor_before])**2 + (y_sensors[i_sensor] - y_columns[i_sensor_before])**2
            distance_on_insoles_before = (insole_activations["position_activation"][1, i_sensor] - sensor_columns[i_sensor_before])**2
            g += [distance_sensors_before - distance_on_insoles_before]
            lbg += [0]
            ubg += [0]

            # The sensors must be on the ellipse
            g += [((x_sensors[i_sensor] - ellipse_center_x) * cos_angle + (
                        y_sensors[i_sensor] - ellipse_center_y) * sin_angle) ** 2 / (ellipse_width / 2.) ** 2 +
                  ((x_sensors[i_sensor] - ellipse_center_x) * sin_angle - (
                              y_sensors[i_sensor] - ellipse_center_y) * cos_angle) ** 2 / (ellipse_height / 2.) ** 2]
            lbg += [1]
            ubg += [1]

    g += [cas.sqrt((x_columns[1:] - x_columns[:-1])**2 + (y_columns[1:] - y_columns[:-1])**2)]
    lbg += [sensor_columns[1:] - sensor_columns[:-1]]
    ubg += [sensor_columns[1:] - sensor_columns[:-1]]

    for i_column in range(len(sensor_columns)):
        # The sensors must be on the ellipse
        g += [((x_columns[i_column] - ellipse_center_x) * cos_angle + (y_columns[i_column] - ellipse_center_y) * sin_angle) ** 2 / (ellipse_width/2.) ** 2 +
              ((x_columns[i_column] - ellipse_center_x) * sin_angle - (y_columns[i_column] - ellipse_center_y) * cos_angle) ** 2 / (ellipse_height/2.) ** 2]
        lbg += [1]
        ubg += [1]

        # Initial guess for the position of the column
        if fig_name[-1] == 'L':
            x0[2*nb_sensors + i_column] = position_markers[0, 0] - i_column * 0.01504
            x0[2*nb_sensors + len(sensor_columns) + i_column] = position_markers[1, 0] + i_column * 0.01504
        elif fig_name[-1] == 'R':
            x0[2*nb_sensors + i_column] = position_markers[0, 0] + i_column * 0.01504
            x0[2*nb_sensors + len(sensor_columns) + i_column] = position_markers[1, 0] + i_column * 0.01504
        else:
            raise RuntimeError("Please contact the lazy dev aka EveCharbie :p")

    nlp = {"x": cas.vertcat(x_sensors, y_sensors, x_columns, y_columns), "f": f, "g": cas.vertcat(*g)}
    opts = {"ipopt.print_level": 5}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)

    sol = solver(x0=x0, lbx=[-np.inf] * (nb_sensors * 2 + len(sensor_columns) * 2), ubx=[np.inf] * (nb_sensors * 2 + len(sensor_columns) * 2), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg))

    if not solver.stats()["success"]:
        x_sensors = x0[:nb_sensors]
        y_sensors = x0[nb_sensors: 2*nb_sensors]
        x_columns = x0[2*nb_sensors:2*nb_sensors+len(sensor_columns)]
        y_columns = x0[2*nb_sensors+len(sensor_columns):]
        cos_angle = cas.cos(np.pi - ellipse_theta)
        sin_angle = cas.sin(np.pi - ellipse_theta)
        for name in names_in_order:
            i_sensor = sensors_and_markers_index[name][0]
            if i_sensor is None:
                continue
            if sensors_and_markers_index[name][1] is not None:
                i_marker = sensors_and_markers_index[name][1]
                print("f += ", (x_sensors[i_sensor] - position_markers[0, i_marker]) ** 2 + (
                            y_sensors[i_sensor] - position_markers[1, i_marker]) ** 2)
            if np.round(insole_activations["position_activation"][1, i_sensor], 7) in np.round(sensor_columns, 7):
                i_column_equal = np.where(
                    np.round(insole_activations["position_activation"][1, i_sensor], 7) == np.round(sensor_columns, 7))[
                    0][0]
                print("g += ", [x_sensors[i_sensor] - x_columns[i_column_equal], y_sensors[i_sensor] - y_columns[i_column_equal]])
            else:
                i_sensor_before = np.where(insole_activations["position_activation"][1, i_sensor] > sensor_columns)[0][
                    -1]
                distance_sensors_before = (x_sensors[i_sensor] - x_columns[i_sensor_before]) ** 2 + (
                            y_sensors[i_sensor] - y_columns[i_sensor_before]) ** 2
                distance_on_insoles_before = (insole_activations["position_activation"][1, i_sensor] - sensor_columns[i_sensor_before]) ** 2
                print("g += ", [distance_sensors_before - distance_on_insoles_before])
                print("g += ", [((x_sensors[i_sensor] - ellipse_center_x) * cos_angle + (y_sensors[i_sensor] - ellipse_center_y) * sin_angle) ** 2 / (ellipse_width / 2.) ** 2 + ((x_sensors[i_sensor] - ellipse_center_x) * sin_angle - (y_sensors[i_sensor] - ellipse_center_y) * cos_angle) ** 2 / (ellipse_height / 2.) ** 2])

        print("g += ", [cas.sqrt((x_columns[1:] - x_columns[:-1]) ** 2 + (y_columns[1:] - y_columns[:-1]) ** 2)])
        for i_column in range(len(sensor_columns)):
            print("g += ", [((x_columns[i_column] - ellipse_center_x) * cos_angle + (
                        y_columns[i_column] - ellipse_center_y) * sin_angle) ** 2 / (ellipse_width / 2.) ** 2 +
                  ((x_columns[i_column] - ellipse_center_x) * sin_angle - (
                              y_columns[i_column] - ellipse_center_y) * cos_angle) ** 2 / (ellipse_height / 2.) ** 2])

        raise RuntimeError(
            "Insole wrapping did not converge, trying again !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    x_opt = sol["x"][:nb_sensors]
    y_opt = sol["x"][nb_sensors:2*nb_sensors]
    x_columns_opt = sol["x"][2*nb_sensors: 2*nb_sensors + len(sensor_columns)]
    y_columns_opt = sol["x"][2*nb_sensors + len(sensor_columns): ]

    distance = []
    for name in names_in_order:
        i_sensor = sensors_and_markers_index[name][0]
        if i_sensor is None:
            continue
        if sensors_and_markers_index[name][1] is not None:
            i_marker = sensors_and_markers_index[name][1]
            distance += [np.abs(np.sqrt((x_opt[i_sensor] - position_markers[0, i_marker]) ** 2 + (
                        y_opt[i_sensor] - position_markers[1, i_marker]) ** 2))]
    print("Mean distance between markers and sensors : ", np.mean(np.array(distance)) * 100, " cm")

    if FLAG_PLOT:
        fig, ax = plt.subplots()
        fig.suptitle("Representation optimization insole")

        # Ellipse
        ellipse = Ellipse(
            xy=(ellipse_center_x, ellipse_center_y),
            width=ellipse_width,
            height=ellipse_height,
            angle=(np.pi - ellipse_theta) * 180 / np.pi,
            facecolor="red",
            alpha=0.5,
        )
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Position markers
        ax.plot(position_markers[0, :], position_markers[1, :], ".b", label="markers from Mocap")

        # Position columns
        ax.plot(x_columns_opt, y_columns_opt, "-g", label="columns of sensors")
        for i_column in range(len(sensor_columns)):
            ax.text(float(x_columns_opt[i_column]), float(y_columns_opt[i_column]), str(i_column), color="g")

        # Position activation
        ax.plot(x_opt, y_opt, ".g", label="activation from marker pushes")

        # Visualisation
        plt.legend()
        plt.savefig(f"Figures/{fig_name}.png")
        plt.close(fig)

    return x_opt, y_opt, x_columns_opt, y_columns_opt, sensor_columns

def get_force_orientation(x_columns_opt, y_columns_opt, ellipse_center_x, ellipse_center_y, FLAG_PLOT):
    """
    Find the unit vector that goes from the sensor columns to the center of the ellipse.
    Since the ellipse is close to a circle, this approximation should be OK.
    Parameters
    ----------
    x_columns_opt
    y_columns_opt
    ellipse_center_x
    ellipse_center_y

    TODO: doc
    Returns
    -------

    """
    force_orientation = np.zeros((x_columns_opt.shape[0], 2))
    for i_column in range(x_columns_opt.shape[0]):
        vector = np.array([x_columns_opt[i_column] - ellipse_center_x, y_columns_opt[i_column] - ellipse_center_y]).reshape(-1, )
        norm = np.linalg.norm(vector)
        force_orientation[i_column, :] = vector / norm

    if FLAG_PLOT:
        fig = plt.figure()
        for i_column in range(x_columns_opt.shape[0]):
            plt.plot(np.array([float(x_columns_opt[i_column]), float(x_columns_opt[i_column]) + force_orientation[i_column, 0]]),
                     np.array([float(y_columns_opt[i_column]), float(y_columns_opt[i_column]) + force_orientation[i_column, 1]]),
                     '-m')
        plt.plot(ellipse_center_x, ellipse_center_y, 'or')
        plt.savefig("Figures/force_orientation.png")
        plt.close(fig)

    return force_orientation

def find_tangent(ellipse_center, ellipse_axes, ellipse_theta, point, fig_name: str, FLAG_PLOT=False):
    """
    Find the tangent of an ellipse at a given point
    Parameters
    ----------
    ellipse_center:
        Center of the ellipse
    ellipse_axes:
        Axes of the ellipse
    ellipse_theta:
        Angle of the ellipse
    point:
        Point at which the tangent is searched
    fig_name:
        Name of the figure
    FLAG_PLOT:
        Flag to plot the ellipse and the tangent

    Returns
    -------
    slope:
        Slope of the tangent
    """

    cx, cy = ellipse_center
    a, b = ellipse_axes
    x, y = point

    # Reverse rotation and reverse translation to obtain point coordinates in the ellipse system
    x1 = (x - cx) * np.cos(ellipse_theta) + (y - cy) * np.sin(ellipse_theta)
    y1 = -(x - cx) * np.sin(ellipse_theta) + (y - cy) * np.cos(ellipse_theta)

    # Coefficients for the tangent equation
    A = x1 / a**2
    B = y1 / b**2

    # Slope of the tangent line in the ellipse's coordinate system
    slope_ellipse = -A / B

    # Convert the slope back to the original coordinate system
    slope = (slope_ellipse * np.cos(ellipse_theta) - np.sin(ellipse_theta)) / (
        np.cos(ellipse_theta) + slope_ellipse * np.sin(ellipse_theta)
    )

    if FLAG_PLOT:
        # Generate points for the tangent line
        t = np.linspace(x - 0.1, x + 0.1, 100)
        tangent_line = y + slope * (t - x)

        # Drawing the ellipse and tangent
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and the tangente of a one marker")

        # Drawing the ellipse
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_theta * 180 / np.pi, facecolor="red", alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Draw the point of intersection
        ax.scatter(x, y, color="blue", label="intersection")

        # Drawing the tangent
        ax.plot(t, tangent_line, color="blue", label="tangente")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        plt.close(fig)

    return slope_ellipse, slope


def find_perpendiculaire_to_tangente(
    tangent_slope, point, ellipse_axes, ellipse_theta, ellipse_center, fig_name: str, FLAG_PLOT=False
):
    """
    Find the perpendicular to the tangent of an ellipse at a given point
    Parameters
    ----------
    tangent_slope:
        Slope of the tangent
    point:
        Point at which the tangent is searched
    ellipse_axes:
        Axes of the ellipse
    ellipse_theta:
        Angle of the ellipse
    ellipse_center:
        Center of the ellipse
    fig_name:
        Name of the figure
    FLAG_PLOT:
        Flag to plot the ellipse and the perpendicular to the tangent

    Returns
    -------
    slope:
        Slope of the perpendicular to the tangent
    """
    x, y = point
    cx, cy = ellipse_center
    a, b = ellipse_axes

    # Slope of the perpendicular to the tangent
    perpendicular_slope = -1 / tangent_slope

    # Generating points for the perpendicular to the tangent
    t = np.linspace(x - 0.1, x + 0.1, 100)
    perpendicular_line = y + perpendicular_slope * (t - x)
    tangent_line = y + tangent_slope * (t - x)

    if FLAG_PLOT:
        # Draw the ellipse, the tangent and the perpendicular to the tangent
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole, tangente and perpendiculaire")

        # Drawing the ellipse
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_theta * 180 / np.pi, facecolor="red", alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect("equal")
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Draw the point of intersection
        ax.scatter(x, y, color="blue", label="intersection")

        # Tracing the tangent
        ax.plot(t, tangent_line, color="blue", label="tagente")

        # Draw the perpendicular to the tangent
        ax.plot(t, perpendicular_line, color="orange", label="perpendiculaire")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        plt.close(fig)

    return perpendicular_line


def position_insole(marker_list: list, model):
    """
    Find the position of the insole
    Parameters
    ----------
    marker_list:
        List of the markers
    model:
        Model of the insole

    Returns
    -------
    center:
        Center of the insole
    position_markers:
        Position of the markers
    """
    if all(isinstance(valeur, int) for valeur in marker_list):
        position_markers = np.zeros(shape=(3, len(marker_list)))
        for i in range(len(marker_list)):
            position_markers[:, i] = model.markers()[marker_list[i]].to_array()
        center = np.mean(position_markers, axis=1)
    else:
        raise RuntimeError("ERROR: The marker list must contain the index of the markers that are in the biorbd model.")
    return center, position_markers

def get_force_from_insoles(insole_data, force_orientation, position_activation, sensor_columns, FLAG_PLOT=True):
    insole_data_array = np.array(insole_data.iloc[3:, 1:-1])
    nb_cells = position_activation["all_sensors_positions"].shape[0]
    force_data_per_cell = np.zeros((insole_data_array.shape[0], insole_data_array.shape[1], 2))
    for i_cell in range(nb_cells):
        position_activation_this_time = position_activation["all_sensors_positions"][i_cell]
        force_orientation_this_time = force_orientation[np.where(position_activation_this_time == sensor_columns), :]
        for i_frame in range(insole_data_array.shape[0]):
            force_data_per_cell[i_frame, i_cell, :] = insole_data_array[i_frame, i_cell] * force_orientation_this_time
    force_data = np.sum(force_data_per_cell, axis=1)

    if FLAG_PLOT:
        fig = plt.figure()
        plt.plot(force_data[:, 0], '-r', label="Force X")
        plt.plot(force_data[:, 1], '-b', label="Force Y")
        plt.legend()
        plt.savefig("Figures/force.png")
        plt.close(fig)

    return force_data