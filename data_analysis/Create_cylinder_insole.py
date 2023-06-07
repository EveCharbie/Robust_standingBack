import numpy as np
from matplotlib.patches import Ellipse
import casadi as cas
from IPython import embed
import biorbd


# --- Functions --- #
def points_to_ellipse_width(centers):
    """
    Optimize the parameters of an ellipse containing 90% of the heatmap points.
    Reports the width and height of the ellipse.
    """

    centers = np.array(centers.T)
    mean_centers = np.mean(centers, axis=0)
    # distance = np.linalg.norm(centers.T - mean_centers[:, np.newaxis], axis=0)
    # distance = (centers - mean_centers[:, np.newaxis]).T
    # distance = abs(centers - mean_centers)
    # percentile = np.percentile(distance, 100)
    # indices_sorted = np.argsort(distance)
    # first_false_index = np.where(np.logical_not(distance[indices_sorted] < percentile))[0]
    # indices = indices_sorted[:first_false_index[0]]

    # Generate an initial guess for the ellipse parameters
    theta_gauss = 0
    width_gauss = 1
    height_gauss = 1

    # State the optimization problem with the following variables
    # Angle (theta)
    # width of the ellipse (b)
    # height of the ellipse (a)
    # x center of the ellipse (xc)
    # y center of the ellipse (yc)
    ellipse_param = cas.MX.sym("parameters", 5)
    x0 = np.array([theta_gauss, width_gauss, height_gauss, mean_centers[0], mean_centers[1]])

    # Objective (minimize squared distance between points and ellipse boundary)
    f = 0
    for i, indices_this_time in enumerate(centers):
        cos_angle = cas.cos(np.pi - ellipse_param[0])
        sin_angle = cas.sin(np.pi - ellipse_param[0])

        xc = centers[indices_this_time, 0] - ellipse_param[3]
        yc = centers[indices_this_time, 1] - ellipse_param[4]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        distance_markers_centre_x = centers[indices_this_time, 0] - ellipse_param[3]
        distance_markers_centre_y = centers[indices_this_time, 1] - ellipse_param[4]

        f += ((xct ** 2 / (ellipse_param[1] / 2.) ** 2) + (yct ** 2 / (ellipse_param[2] / 2.) ** 2) - (cas.sqrt(distance_markers_centre_x ** 2 + distance_markers_centre_y ** 2))) ** 2

    nlp = {"x": ellipse_param, "f": f}
    opts = {"ipopt.print_level": 5}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=[-np.pi, 0, 0, mean_centers[:, 0] - 0.5, mean_centers[:, 1] - 0.5],
                 ubx=[np.pi, 0.2, 0.3, mean_centers[:, 0] + 0.5, mean_centers[:, 1] + 0.5])

    if solver.stats()["success"]:
        success_out = True
        theta_opt = sol["x"][0]
        width_opt = sol["x"][1]
        height_opt = sol["x"][2]
        center_x_opt = sol["x"][3]
        center_y_opt = sol["x"][4]
    else:
        print("Ellipse did not converge, trying again")

    print("Paramètres optimaux de l'ellipse: ")
    print("Hauteur de l'ellipse: " + str(height_opt))
    print("Largeur de l'ellipse: " + str(width_opt))
    print("Centre x de l'ellipse: " + str(center_x_opt))
    print("Centre y de l'ellipse: " + str(center_y_opt))
    print("Angle theta de l'ellipse: " + str(center_y_opt))
    return width_opt, height_opt, center_x_opt, center_y_opt, theta_opt


def position_insole(marker_list: list):
    """
    Find center of the insole

    marker_list: List marker insole with only index of markers in the model
    position_markers: Return the position of all markers of the insole
    center: Return center insole

    """
    if all(isinstance(valeur, int) for valeur in marker_list):
        position_markers = np.zeros(shape=(3, len(marker_list)))
        for i in range(len(marker_list)):
            position_markers[:, i] = model.markers()[marker_list[i]].to_array()
        center = np.mean(position_markers, axis=1)
    else:
        print("ERROR: La liste doit contenir les index des markers dans le modèle !")
    return center, position_markers

# --- Parameters --- #
model_path = "EmCo.bioMod"
model = biorbd.Model(model_path)
markers_list = []
for i in range(model.nbMarkers()):
    markers_list.append(model.markerNames()[i].to_string())

# --- Script --- #
# Markers insole L
    # Center insole L
markers_insole_L = [index for index, markers in enumerate(markers_list) if "G" in markers and "insole" in markers]
center_L, position_markers_L = position_insole(markers_insole_L)

    # Ellipse insole L
print("Insole Gauche")
distance_L, width_ellipse_L, height_ellipse_L = points_to_ellipse_width(position_markers_L[:2, :])

# Markers insole R
    # Center insole R
markers_insole_R = [index for index, markers in enumerate(markers_list) if "D" in markers and "insole" in markers]
center_R, position_markers_R = position_insole(markers_insole_R)

    # Ellipse insole L
print("Insole Droite")
distance_R, width_ellipse_R, height_ellipse_R = points_to_ellipse_width(position_markers_R[:2, :])
