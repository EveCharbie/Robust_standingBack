import casadi as cas
from IPython import embed
import biorbd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from copy import copy
from scipy.optimize import fsolve
import math
from cartography_insoles import position_activation


# --- Functions --- #

def norm_2D(pointA, pointB) -> float:
    """
    Find the distance between two points
    :param pointA: A point with a coordinate on x and y
    :param pointB: A other point with a coordinate on x and y
    :return: The distance between this two points
    """
    norm = np.sqrt(
        ((pointA[0] - pointB[0]) ** 2) + ((pointA[1] - pointB[1]) ** 2)
    )
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

        eq1 = (x * np.cos(theta) + y * np.sin(theta)) ** 2 / a ** 2 + (
                y * np.cos(theta) - x * np.sin(theta)) ** 2 / b ** 2 - 1
        eq2 = y - y1 - m * (x - x1)
        return [eq1, eq2]

    # Utiliser fsolve de scipy pour trouver les solutions
    x, y = fsolve(equations, (x1, y1))
    intersection = x + h, y + k

    if FLAG_PLOT:
        ellipse = Ellipse(xy=(h, k),
                          width=a, height=b,
                          angle=theta * 180 / np.pi, facecolor='orange', alpha=0.5)

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
        plt.plot(x_values, y_values, label='Droite')
        ax.scatter(h, k, color='red', label='centre ellipse')
        ax.scatter(x1, y1, color='orange', label='markers')
        ax.scatter(intersection[0], intersection[1], color='blue', label='intersection')
        ax.add_patch(ellipse)
        ax.set_aspect('equal')
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
    x = cas.MX.sym('x')

    # Calcule de la pente (m)
    m = (yB - yA) / (xB - xA)

    # Calcule de l'ordonnée
    b = yA - m * xA

    # Equation de la droite
    y = m * x + b

    # Création fonction casadi
    func = cas.Function('f', [x], [y])
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
        norm.append(norm_2D(data[index_marker_parameter["marker_" + str(index_doublon[i])][0], :],
                            data[index_marker_parameter["marker_" + str(index_doublon[i])][1], :]))
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
        [index_marker_parameter["marker_up"] + index_marker_parameter[
             "marker_down"], "up_down"]]

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

            f += ((xct ** 2 / (((ellipse_param[1]) / 2) ** 2)) + (yct ** 2 / (((ellipse_param[2]) / 2) ** 2))
                  - cas.sqrt(xc ** 2 + yc ** 2)) ** 2


        nlp = {"x": ellipse_param, "f": f}
        opts = {"ipopt.print_level": 5}
        solver = cas.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=x0, lbx=[-np.pi, 0, 0, mean_centers[0] - 0.5, mean_centers[1] - 0.5],
                     ubx=[np.pi, 0.2, 0.3, mean_centers[0] + 0.5, mean_centers[1] + 0.5])

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
        ellipse_up = Ellipse(xy=(ellipse[0]["center_x_ellipse"], ellipse[0]["center_y_ellipse"]),
                             width=ellipse[0]["a"]/2, height=ellipse[0]["b"]/2,
                             angle=ellipse[0]["angle"]*180/np.pi, facecolor='red', alpha=0.5)
        ellipse_down = Ellipse(xy=(ellipse[1]["center_x_ellipse"], ellipse[1]["center_y_ellipse"]),
                               width=ellipse[1]["a"]/2, height=ellipse[1]["b"]/2,
                               angle=ellipse[1]["angle"]*180/np.pi, facecolor='blue', alpha=0.5)
        ellipse_all = Ellipse(xy=(ellipse[2]["center_x_ellipse"], ellipse[2]["center_y_ellipse"]),
                              width=ellipse[2]["a"]/2, height=ellipse[2]["b"]/2,
                              angle=ellipse[2]["angle"]*180/np.pi, facecolor='orange', alpha=0.5)

        # Integration markers
        up_markers = ax.plot(data[index_marker_parameter["marker_up"], 0],
                             data[index_marker_parameter["marker_up"], 1], 'ro', label="markers up")
        down_markers = ax.plot(data[index_marker_parameter["marker_down"], 0],
                               data[index_marker_parameter["marker_down"], 1], 'bo', label="markers down")
        mid_markers = ax.plot(data[index_marker_parameter["marker_mid"], 0],
                              data[index_marker_parameter["marker_mid"], 1], 'go', label="markers mid")


        # Ajout de l'ellipse aux axes
        ax.add_patch(ellipse_up)
        ax.add_patch(ellipse_down)
        ax.add_patch(ellipse_all)
        ax.set_aspect('equal')

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


def minimize_distance(position_markers, ellipse_center, ellipse_axes, ellipse_angle):
    # Position of the markers, center of the ellipse, axes of the ellipse, angle of the ellipse
    x, y = position_markers[0], position_markers[1]
    xc, yc = ellipse_center[0], ellipse_center[1]
    a, b = ellipse_axes[0], ellipse_axes[1]
    theta = ellipse_angle

    # Generate an initial guess for the optimization variables
    pos = cas.MX.sym("pos", 2)

    f = 0
    # Objective (minimize distance between two points)
    for markers in range(len(x)):
        # f = cas.sqrt((x[0] - pos[0])**2 + (y[0] - pos[0])**2)
        distance_marker_sensor = cas.sqrt((x[markers] - pos[0])**2 + (y[markers] - pos[1])**2)
        distance_marker_center_ellipse = cas.sqrt((x[markers] - xc)**2 + (y[markers] - yc)**2)
        f += cas.sin((distance_marker_sensor / distance_marker_center_ellipse))

    nlp = {"x": pos, "f": f}
    opts = {"ipopt.print_level": 5}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)
    x0 = np.zeros(2)  # Initial guess for the optimization variables

    sol = solver(x0=x0, lbx=[-np.inf, -np.inf], ubx=[np.inf, np.inf])

    if solver.stats()["success"]:
        angle = float(sol["f"])
        print("Angle en radian:", angle)
        return angle
    else:
        print("Optimization did not converge")
        return None



def find_tangent(ellipse_center, ellipse_axes, ellipse_angle, point, fig_name:str,  FLAG_PLOT=False):
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
    A = x1 / a ** 2
    B = y1 / b ** 2

    # Slope of the tangent line in the ellipse's coordinate system
    slope_ellipse = -A / B

    # Convert the slope back to the original coordinate system
    slope = (slope_ellipse * np.cos(ellipse_angle) - np.sin(ellipse_angle)) / (np.cos(ellipse_angle) + slope_ellipse * np.sin(ellipse_angle))

    if FLAG_PLOT:
        # Generate points for the tangent line
        t = np.linspace(x - 0.1, x + 0.1, 100)
        tangent_line = y + slope * (t - x)

        # Tracer l'ellipse et la tangente
        fig, ax = plt.subplots()
        fig.suptitle("Representation insole and the tangente of a one marker")

        # Tracer l'ellipse
        ellipse = Ellipse(xy=(cx, cy),
                             width=a, height=b,
                             angle=ellipse_angle * 180 / np.pi, facecolor='red', alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect('equal')
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Tracer le point d'intersection
        ax.scatter(x, y, color='blue', label='intersection')

        # Tracer la tangente
        ax.plot(t, tangent_line, color="blue", label="tangente")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        # plt.show()

    return slope_ellipse, slope


def find_perpendiculaire_to_tangente(tangent_slope, point, ellipse_axes, ellipse_angle, ellipse_center, fig_name:str, FLAG_PLOT=False):
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
        ellipse = Ellipse(xy=(cx, cy),
                          width=a, height=b,
                          angle=ellipse_angle * 180 / np.pi, facecolor='red', alpha=0.5)
        ax.add_patch(ellipse)
        ax.set_aspect('equal')
        ax.set_xlabel(" Position in x")
        ax.set_ylabel(" Position in y")

        # Tracer le point d'intersection
        ax.scatter(x, y, color='blue', label='intersection')

        # Tracer la tangente
        ax.plot(t, tangent_line, color="blue", label="tagente")

        # Tracer la perpendiculaire à la tangente
        ax.plot(t, perpendicular_line, color="orange", label="perpendiculaire")
        plt.legend()
        plt.savefig("Figures/" + fig_name + ".svg")
        fig.clf()
        plt.show()

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
