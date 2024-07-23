"""
Main code calling scipt python create_cylinder_insole functions to analyze pressure inserts data
"""

import pandas as pd
import numpy as np
import biorbd
from matplotlib.patches import Ellipse
from Create_cylinder_insole import (
    equation_droite,
    position_insole,
    points_to_ellipse,
    position_activation,
    find_tangent,
    find_perpendiculaire_to_tangente,
    find_index_by_name,
    intersection_ellipse_line,
    minimize_distance,
    norm_2D,
    change_ref_marker,
    distance_between_line_sensors,
    cartography_insole,
    lissage,
)

# --- Parameters --- #
model_path = "EmCo.bioMod"
model = biorbd.Model(model_path)
markers_list = []
for i in range(model.nbMarkers()):
    markers_list.append(model.markerNames()[i].to_string())

# --- Markers insole Left --- #
markers_insole_L = [index for index, markers in enumerate(markers_list) if "G" in markers and "insole" in markers]
markers_insole_L_name = [
    markers for index, markers in enumerate(markers_list) if "G" in markers and "insole" in markers
]
center_L, position_markers_L = position_insole(markers_insole_L, model)
markers_insole_L_xy = position_markers_L[:2, :]


# Cartography insole
# Load info_insole, insole_R, insole_L
markers_insole_R = pd.read_csv(
    "/home/lim/Anais/CollecteStandingBack/EmCo_insoles_rename/markers_insoles_R_1_L.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)
markers_insole_L = pd.read_csv(
    "/home/lim/Anais/CollecteStandingBack/EmCo_insoles_rename/markers_insoles_L_1_R.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)
insoles_coordonnees = pd.read_csv(
    "/home/lim/Anais/CollecteStandingBack/Access_sensor_positions/coordonnees_insoles.csv",
    sep=";",
    decimal=",",
    low_memory=False,
    header=0,
    na_values="NaN",
)

markers_insole_R, markers_insole_L = markers_insole_R.iloc[3:, 1:-1], markers_insole_L.iloc[3:, 1:-1]
markers_insole_R.reset_index(drop=True, inplace=True)
markers_insole_L.reset_index(drop=True, inplace=True)
sensor_45 = markers_insole_R.columns.to_list()  # List sensor on the insole size 45 (without columns Sync and Time)
coordonnees_insole_45 = insoles_coordonnees.loc[0:7, sensor_45]

# Cartography insole
cartography_insole(
    file_insole=markers_insole_R, file_info_insole=coordonnees_insole_45, fig_name="cartography", FLAG_PLOT=True
)


# Find ellipse
parameters_ellipse_L = points_to_ellipse(
    data=markers_insole_L_xy, fig_name="Ellipse_insole_L", markers_name=markers_insole_L_name, FLAG_PLOT=False
)
center_ellipse_L = np.array([parameters_ellipse_L[2]["center_x_ellipse"], parameters_ellipse_L[2]["center_y_ellipse"]])

# TODO: Further optimization to fit cells (minimize error position (position sensor - position marker)
#  TODO: + keep sensors at a constant distance from markers

# Position of activations in x and y
position_activation_R, position_activation_L = position_activation(
    file_insole_R=markers_insole_R,
    file_insole_L=markers_insole_L,
    file_info_insole=coordonnees_insole_45,
    FLAG_PLOT=True,
)

# Second optimisation
distance_opti_L = minimize_distance(
    position_markers=markers_insole_L_xy,
    position_activation=position_activation_L["activation"],
    coordonnees_insole_45_y=position_activation_L["distance_sensor_y"],
    ellipse_center=center_ellipse_L,
    ellipse_axes=(parameters_ellipse_L[2]["a"], parameters_ellipse_L[2]["b"]),
    ellipse_angle=parameters_ellipse_L[2]["angle"],
)

# TODO: Vector sum of activations

# Find line, tangente and perpendiculaire
droite_tangente_perpendiculaire_L = []
ellipse_L = Ellipse(
    xy=(parameters_ellipse_L[2]["center_x_ellipse"], parameters_ellipse_L[2]["center_y_ellipse"]),
    width=parameters_ellipse_L[2]["a"],
    height=parameters_ellipse_L[2]["b"],
    angle=parameters_ellipse_L[2]["angle"] * 180 / np.pi,
    facecolor="orange",
    alpha=0.5,
)

for i in range(markers_insole_L_xy.shape[1]):
    func_droite_L = equation_droite(markers_insole_L_xy[:, i], center_ellipse_L)
    intersection = intersection_ellipse_line(
        line_points=(markers_insole_L_xy[:, i], center_ellipse_L),
        ellipse_center=center_ellipse_L,
        a=parameters_ellipse_L[2]["a"] / 2,
        b=parameters_ellipse_L[2]["b"] / 2,
        theta=parameters_ellipse_L[2]["angle"],
        FLAG_PLOT=False,
    )

    tangente = find_tangent(
        ellipse_center=center_ellipse_L,
        ellipse_axes=(parameters_ellipse_L[2]["a"], parameters_ellipse_L[2]["b"]),
        ellipse_angle=parameters_ellipse_L[2]["angle"],
        point=intersection,
        FLAG_PLOT=False,
        fig_name="Tangente_insole_L_" + str(i),
    )
    perpendiculaire = find_perpendiculaire_to_tangente(
        ellipse_center=center_ellipse_L,
        ellipse_axes=(parameters_ellipse_L[2]["a"], parameters_ellipse_L[2]["b"]),
        ellipse_angle=parameters_ellipse_L[2]["angle"],
        point=intersection,
        tangent_slope=tangente[0],
        FLAG_PLOT=False,
        fig_name="Perpendiculaire_insole_L_" + str(i),
    )

    # Ranges of x values for tracing
    pos_x = abs(parameters_ellipse_L[2]["center_x_ellipse"] - markers_insole_L_xy[0, i])
    x_values = np.linspace(-pos_x - 0.01, pos_x + 0.01, 100)

    # Evaluation of y for each value of x
    y_values = [float(func_droite_L(x).full()[0]) for x in x_values]
    droite_tangente_perpendiculaire_L.append([(x_values, y_values), intersection, tangente])


# --- Markers insole Right --- #
markers_insole_R = [index for index, markers in enumerate(markers_list) if "D" in markers and "insole" in markers]
markers_insole_R_name = [
    markers for index, markers in enumerate(markers_list) if "D" in markers and "insole" in markers
]
center_R, position_markers_R = position_insole(markers_insole_R, model)
markers_insole_R_xy = position_markers_R[:2, :]

# Find ellipse
parameters_ellipse_R = points_to_ellipse(
    data=position_markers_R[:2, :], fig_name="Ellipse_insole_R", markers_name=markers_insole_R_name, FLAG_PLOT=False
)
center_ellipse_R = np.array([parameters_ellipse_R[2]["center_x_ellipse"], parameters_ellipse_R[2]["center_y_ellipse"]])

# TODO: Further optimization to fit cells (minimize error position (position sensor - position marker)
#  TODO: + keep sensors at a constant distance from markers

# TODO: Vector sum of activations

# Find line, tangente and perpendiculaire
droite_tangente_perpendiculaire_R = []
ellipse_R = Ellipse(
    xy=(parameters_ellipse_R[2]["center_x_ellipse"], parameters_ellipse_R[2]["center_y_ellipse"]),
    width=parameters_ellipse_R[2]["a"],
    height=parameters_ellipse_R[2]["b"],
    angle=parameters_ellipse_R[2]["angle"] * 180 / np.pi,
    facecolor="orange",
    alpha=0.5,
)

for i in range(markers_insole_R_xy.shape[1]):
    func_droite_R = equation_droite(markers_insole_R_xy[:, i], center_ellipse_R)
    intersection = intersection_ellipse_line(
        line_points=(markers_insole_R_xy[:, i], center_ellipse_R),
        ellipse_center=center_ellipse_R,
        a=parameters_ellipse_R[2]["a"] / 2,
        b=parameters_ellipse_R[2]["b"] / 2,
        theta=parameters_ellipse_R[2]["angle"],
    )

    tangente = find_tangent(
        ellipse_center=center_ellipse_R,
        ellipse_axes=(parameters_ellipse_R[2]["a"], parameters_ellipse_R[2]["b"]),
        ellipse_angle=parameters_ellipse_R[2]["angle"],
        point=intersection,
        FLAG_PLOT=True,
        fig_name="Tangente_insole_R_" + str(i),
    )
    perpendiculaire = find_perpendiculaire_to_tangente(
        ellipse_center=center_ellipse_R,
        ellipse_axes=(parameters_ellipse_R[2]["a"], parameters_ellipse_R[2]["b"]),
        ellipse_angle=parameters_ellipse_R[2]["angle"],
        point=intersection,
        tangent_slope=tangente[0],
        FLAG_PLOT=True,
        fig_name="Perpendiculaire_insole_R_" + str(i),
    )

    # Ranges of x values for tracing
    pos_x = abs(parameters_ellipse_R[2]["center_x_ellipse"] - markers_insole_R_xy[0, i])
    x_values = np.linspace(-pos_x - 0.01, pos_x + 0.01, 100)

    # Evaluation of y for each value of x
    y_values = [float(func_droite_R(x).full()[0]) for x in x_values]
    droite_tangente_perpendiculaire_R.append([(x_values, y_values), intersection, tangente])
