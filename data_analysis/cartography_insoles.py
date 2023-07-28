import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from Create_cylinder_insole import distance_between_line_sensors


# --- Functions --- #
def find_activation_sensor(distance_sensor_y, position_activation_y):
    """
    Find the information of the sensor activated
    Parameters
    ----------
    distance_sensor_y:
        Distance between the sensor and the activation in y
    position_activation_y:
        Position of the activation in y

    Returns
    -------

    """
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
    """
    Plot the cartography of the insoles
    Parameters
    ----------
    file_insole:
        File of an insole
    file_info_insole:
        File with the information of the insole (coordinate sensor insole)
    fig_name:
        Name of the figure
    FLAG_PLOT:
        if you want the plot related to the cartography of the insole

    Returns
    -------
    Plot the cartography of the insoles
    """

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
