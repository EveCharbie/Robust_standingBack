"""
Main code calling scipt python create_cylinder_insole functions to analyze pressure inserts data.
WARNING: the right foot insole was placed on the left tibia and inversely.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import biorbd
import pickle
from Create_cylinder_insole import (
    position_insole,
    points_to_circle,
    position_activation,
    minimize_distance_circle,
    cartography_insole,
    get_force_orientation,
    get_force_from_insoles,
)

model_path = "EmCo.bioMod"
data_folder = "EmCo_insoles_rename/"

# --- Load the biorbd model that will be used for kinematic reconstructions --- #
model = biorbd.Model(model_path)
markers_list = []
for i in range(model.nbMarkers()):
    markers_list.append(model.markerNames()[i].to_string())
# Find the markers that are on the right and left insoles in the model
markers_insole_L_idx = [index for index, markers in enumerate(markers_list) if "G" in markers and "insole" in markers]
markers_insole_L_name = [
    markers for index, markers in enumerate(markers_list) if "G" in markers and "insole" in markers
]
markers_insole_R_idx = [index for index, markers in enumerate(markers_list) if "D" in markers and "insole" in markers]
markers_insole_R_name = [
    markers for index, markers in enumerate(markers_list) if "D" in markers and "insole" in markers
]

# # Show the model
# import bioviz
# b = bioviz.Viz(model_path)
# b.exec()

center_L, position_markers_L = position_insole(markers_insole_L_idx, model)
center_R, position_markers_R = position_insole(markers_insole_R_idx, model)

#Here we consider that the tibia was up right during the calibration trial, so we can take only the x an y componenets and drop the z component.
markers_insole_L_xy = position_markers_L[:2, :]
markers_insole_R_xy = position_markers_R[:2, :]

# Cartography insole
# Load insole_R: calibration file for the right insole (pressure data).
# TODO: @anaisfarr: please confirm the order
insole_calibration_data_R = pd.read_csv(
    data_folder + "markers_insoles_R_1_L.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)
# Load insole_L: calibration file for the left insole (pressure data).
insole_calibration_data_L = pd.read_csv(
    data_folder + "markers_insoles_L_1_R.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)
# Load coordonnees_insoles: manufacturer file that link the number of the sensor with its coordinates on the insole.
insoles_coordonnees = pd.read_csv(
    data_folder + "coordonnees_insoles.csv",
    sep=";",
    decimal=",",
    low_memory=False,
    header=0,
    na_values="NaN",
)

# Skipping the three first frames and exclude the first and last sensors sice they are outside the insole for this foot size
insole_calibration_data_R, insole_calibration_data_L = insole_calibration_data_R.iloc[3:, 1:-1], insole_calibration_data_L.iloc[3:, 1:-1]
insole_calibration_data_R.reset_index(drop=True, inplace=True)
insole_calibration_data_L.reset_index(drop=True, inplace=True)
existing_sensor_index = insole_calibration_data_R.columns.to_list()  # List sensor on the insole size 45 (without columns Sync and Time)

insole_coordinates = insoles_coordonnees.loc[0:7, existing_sensor_index]

# Cartography insole
cartography_insole(
    file_insole=insole_calibration_data_R, file_info_insole=insole_coordinates, FLAG_PLOT=True
)

# Find circle
parameters_circle_L = points_to_circle(
    markers_insole_xy=markers_insole_L_xy, fig_name="Circle_insole_L", markers_name=markers_insole_L_name, FLAG_PLOT=True
)
parameters_circle_R = points_to_circle(
    markers_insole_xy=markers_insole_R_xy, fig_name="Circle_insole_R", markers_name=markers_insole_R_name, FLAG_PLOT=True
)

# Position of activations in x and y
position_activation_R, position_activation_L = position_activation(
    file_insole_R=insole_calibration_data_R,
    file_insole_L=insole_calibration_data_L,
    file_info_insole=insole_coordinates,
    FLAG_PLOT=True,
)

# Wrap the insoles optimally on the tibia cylinder
x_opt_L, y_opt_L, x_columns_opt_L, y_columns_opt_L, sensor_columns_L = minimize_distance_circle(
    position_markers=markers_insole_L_xy,
    markers_insole_name=markers_insole_L_name,
    insole_activations=position_activation_L,
    circle_radius=parameters_circle_L[1]["radius"],
    circle_center_x=parameters_circle_L[1]["center_x"],
    circle_center_y=parameters_circle_L[1]["center_y"],
    type='down',
    fig_name="fit_activations_on_circle_L"
)
x_opt_R, y_opt_R, x_columns_opt_R, y_columns_opt_R, sensor_columns_R = minimize_distance_circle(
    position_markers=markers_insole_R_xy,
    markers_insole_name=markers_insole_R_name,
    insole_activations=position_activation_R,
    circle_radius=parameters_circle_R[0]["radius"],
    circle_center_x=parameters_circle_R[0]["center_x"],
    circle_center_y=parameters_circle_R[0]["center_y"],
    type='up',
    fig_name="fit_activations_on_circle_R"
)

# Get the orientation of the force for each insole sensor column
force_orientation_L = get_force_orientation(x_columns_opt=x_columns_opt_L,
                                          y_columns_opt=y_columns_opt_L,
                                          center_x=parameters_circle_L[1]["center_x"],
                                          center_y=parameters_circle_L[1]["center_y"],
                                          FLAG_PLOT=True)
force_orientation_R = get_force_orientation(x_columns_opt=x_columns_opt_R,
                                          y_columns_opt=y_columns_opt_R,
                                          center_x=parameters_circle_R[0]["center_x"],
                                          center_y=parameters_circle_R[0]["center_y"],
                                          FLAG_PLOT=True)

# Find the knee markers
Q = np.zeros((model.nbQ(), ))
markers_knee_L_idx = [markers_list.index('CONDEXTG'), markers_list.index('CONDINTG')]
markers_knee_R_idx = [markers_list.index('CONDEXTD'), markers_list.index('CONDINTD')]
markers_knee_L_mean = np.mean(np.array([model.markers(Q)[markers_knee_L_idx[0]].to_array(), model.markers(Q)[markers_knee_L_idx[1]].to_array()]), axis=0)
markers_knee_R_mean = np.mean(np.array([model.markers(Q)[markers_knee_R_idx[0]].to_array(), model.markers(Q)[markers_knee_R_idx[1]].to_array()]), axis=0)
insoles_markers_L = np.array([model.markers(Q)[idx].to_array() for idx in markers_insole_L_idx])
insoles_markers_R = np.array([model.markers(Q)[idx].to_array() for idx in markers_insole_R_idx])
markers_insole_L_mean = np.mean(insoles_markers_L, axis=0)
markers_insole_R_mean = np.mean(insoles_markers_R, axis=0)
print("The distance between the mean left knee markers and mean insoles markers is : ", np.linalg.norm(markers_insole_L_mean - markers_knee_L_mean), " m")
print("The distance between the mean right knee markers and mean insoles markers is : ", np.linalg.norm(markers_insole_R_mean - markers_knee_R_mean), " m")


for filename in os.listdir(data_folder):
    if "salto" in filename and (filename.split('_')[-1] == 'L.CSV' or filename.split('_')[-1] == 'R.CSV'):
        insole_data = pd.read_csv(
            data_folder + "/" + filename,
            sep=",",
            decimal=".",
            low_memory=False,
            header=0,
            na_values="NaN",
        )
        if filename[-5] == 'L':
            force_data = get_force_from_insoles(insole_data=insole_data,
                                                force_orientation=force_orientation_L,
                                                position_activation=position_activation_L,
                                                sensor_columns=sensor_columns_L,
                                                FLAG_PLOT=True)
        elif filename[-5] == 'R':
            force_data = get_force_from_insoles(insole_data=insole_data,
                                                force_orientation=force_orientation_R,
                                                position_activation=position_activation_R,
                                                sensor_columns=sensor_columns_R,
                                                FLAG_PLOT=True)
        else:
            raise RuntimeError("Wrong file name, it should end with L.CSV or R.CSV")

        with open(f"EmCo_insoles_forces/{filename[:-4]}.pkl", 'wb') as file:
            pickle.dump(force_data, file)









