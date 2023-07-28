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


def cartography_insole(file_insole, file_info_insole, fig_name:str, FLAG_PLOT=False):
    """
    Plot the cartography of the insoles
    Parameters
    ----------
    file_insole:
        File of the insole
    file_info_insole:
        File of the information of the insole
    fig_name:
        Name of the figure
    FLAG_PLOT:
        Flag to plot the results

    Returns
    -------

    """

    sensor_45 = file_insole.columns[1:-1].to_list()     #  List sensor on the insole size 45 (without columns Sync and Time)
    coordonnees_insole_45 = file_info_insole.loc[0:7, sensor_45]

    # Plot : Insoles
    if FLAG_PLOT:
        fig, axs = plt.subplots(2)
        fig.suptitle("Insoles cartography")

            # Subplot 1 : insoles R
        axs[0].plot(coordonnees_insole_45.iloc[7, :], coordonnees_insole_45.iloc[6, :], 'ro')
        axs[0].set_ylabel("Position Y (mm)", fontsize=14)
        axs[0].title.set_text('Insole Right')

            # Subplot 2 : insoles L
        axs[1].plot(coordonnees_insole_45.iloc[5, :], coordonnees_insole_45.iloc[4, :], 'ro')
        axs[1].set_xlabel("Position X (mm)", fontsize=14)
        axs[1].set_ylabel("Position Y (mm)", fontsize=14)
        axs[1].title.set_text('Insole Left')
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()


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


def points_to_ellipse(data, fig_name, markers_name, FLAG_PLOT: False) -> list:
    """
    Find the ellipse parameters from the markers
    Parameters
    ----------
    data:
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
    # TODO: Takes into account markers in front of the knee

    centers_index = [
        [index_marker_parameter["marker_up"], "up"],
        [index_marker_parameter["marker_down"], "down"],
        [index_marker_parameter["marker_up"] + index_marker_parameter["marker_down"], "up_down"],
    ]

    ellipse = []

    for i in range(len(centers_index)):
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
        plt.savefig(fig_name + ".svg")
        plt.show()
        plt.close(plt)

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
    Minimize the distance between the markers and the ellipse
    Parameters
    ----------
    position_markers:
        Position of the markers in the global frame
    position_activation:
        Position of the activation markers in the global frame
    coordonnees_insole_45_y:
        Position of the sensors on the insole
    ellipse_center:
        Center of the ellipse
    ellipse_axes:
        Axes of the ellipse
    ellipse_angle:
        Angle of the ellipse
    FLAG_PLOT:
        Flag to plot the results

    Returns
    -------

    """
    # Change insole ref to put the zero on the first line
    x, y = change_ref_marker(position_markers[0]), change_ref_marker(position_markers[1])

    # Compute the "Y position"on the sensors on the insole
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
            ((xct**2 / (((ellipse_axes[0]) / 2) ** 2))
            + (yct**2 / (((ellipse_axes[1]) / 2) ** 2)))**2
            - (xc_0**2 + yc_0**2)
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
                ((xct**2 / (((ellipse_axes[0]) / 2) ** 2))
                + (yct**2 / (((ellipse_axes[1]) / 2) ** 2)))**2
                - (xc**2 + yc**2)
            )
            ** 2
        ]
        lbg += [1]
        ubg += [1]

        # TODO: make sure it goes in the right direction
        tata = cas.MX.zeros(2)
        tata[0] = xc - xc_0
        tata[1] = yc - yc_0
        g += [(insole_sensors[i_sensor] - insole_sensors[i_sensor - 1])**2 - (tata[0] - tata[1])**2]
        lbg += [0]
        ubg += [0]

        xc_0 = xc
        yc_0 = yc

    markers = 0
    activation_position = cas.MX.zeros(2)
    for i in range(len(position_activation["index_pre"])):
        activation_position[0] = x_sensors[position_activation["index_pre"][i]] + position_activation["pourcentage"][
            i
        ] * (x_sensors[position_activation["index_post"][i]] - x_sensors[position_activation["index_pre"][i]])
        activation_position[1] = y_sensors[position_activation["index_pre"][i]] + position_activation["pourcentage"][
            i
        ] * (y_sensors[position_activation["index_post"][i]] - y_sensors[position_activation["index_pre"][i]])
        distance_marker_sensor = (x[markers] - activation_position[0]) ** 2 + (y[markers] - activation_position[1]) ** 2
        f += distance_marker_sensor

    nlp = {"x": cas.vertcat(angle_to_put_zero, x_sensors, y_sensors), "f": f, "g": cas.vertcat(*g)}
    opts = {"ipopt.print_level": 5}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)
    x0 = np.zeros((18 * 2,))  # Initial guess for the optimization variables

    sol = solver(x0=x0, lbx=[-np.inf] * 18 * 2, ubx=[np.inf] * 18 * 2, lbg=lbg, ubg=ubg)

    fig, ax = plt.subplots()
    fig.suptitle("Representation optimization insole")

    # Ellipse
    ellipse = Ellipse(xy=(ellipse_center[0], ellipse_center[1]), width=ellipse_axes[0]/2, height=ellipse_axes[1]/2, angle=ellipse_angle * 180 / np.pi, facecolor="red", alpha=0.5)
    ax.add_patch(ellipse)
    ax.set_aspect("equal")
    ax.set_xlabel(" Position in x")
    ax.set_ylabel(" Position in y")

    # Position markers
    ax.scatter(position_markers[0, :], position_markers[1, :], color="blue", label="markers")

    # Position activation
    # ax.scatter(x, y, color="green", label="activation")

    # Visualisation
    plt.show()
    plt.close(fig)

    if solver.stats()["success"]:
        output_variables = float(sol["x"])
        print("output_variables", output_variables)
        return output_variables
    else:
        print("Optimization did not converge")
        return None


def find_tangent(ellipse_center, ellipse_axes, ellipse_angle, point, fig_name: str, FLAG_PLOT=False):
    """
    Find the tangent of an ellipse at a given point
    Parameters
    ----------
    ellipse_center:
        Center of the ellipse
    ellipse_axes:
        Axes of the ellipse
    ellipse_angle:
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
    x1 = (x - cx) * np.cos(ellipse_angle) + (y - cy) * np.sin(ellipse_angle)
    y1 = -(x - cx) * np.sin(ellipse_angle) + (y - cy) * np.cos(ellipse_angle)

    # Coefficients for the tangent equation
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

        # Drawing the ellipse and tangent
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and the tangente of a one marker")

        # Drawing the ellipse
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_angle * 180 / np.pi, facecolor="red", alpha=0.5)
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
        plt.show()

    return slope_ellipse, slope


def find_perpendiculaire_to_tangente(
    tangent_slope, point, ellipse_axes, ellipse_angle, ellipse_center, fig_name: str, FLAG_PLOT=False
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
    ellipse_angle:
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
        ellipse = Ellipse(xy=(cx, cy), width=a, height=b, angle=ellipse_angle * 180 / np.pi, facecolor="red", alpha=0.5)
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
        plt.show()

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
        position_markers = None
        center = None
        print("ERROR: La liste doit contenir les index des markers dans le modèle !")
    return center, position_markers
