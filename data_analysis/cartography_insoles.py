import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal


# --- Functions --- #
def lissage(signal_brut, L):
    res = np.copy(signal_brut)  # duplication des valeurs
    for i in range(1, len(signal_brut) - 1):  # toutes les valeurs sauf la première et la dernière
        L_g = min(i, L)  # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut)-i-1, L) # nombre de valeurs disponibles à droite
        Li = min(L_g, L_d)
        res[i] = np.sum(signal_brut[i - Li:i + Li + 1]) / (2 * Li + 1)
    return res


# --- Parameters --- #
SAVE_CARTOGRAPHY = False
FLAG_PLOT = True

# --- Cartography insoles --- #
info_insole_folder = "/home/lim/Anais/CollecteStandingBack/Access_sensor_positions"
trial_insole_folder = "/home/lim/Anais/CollecteStandingBack/EmCo_insoles_rename"

insoles_coordonnees = pd.read_csv(
    str(info_insole_folder) + "/coordonnees_insoles.csv",
    sep=";",
    decimal=",",
    low_memory=False,
    header=0,
    na_values="NaN",
)

insole_R = pd.read_csv(
    str(trial_insole_folder) + "/salto_control_pre_1_R.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)

sensor_45 = insole_R.columns[1:-1].to_list()     #  List sensor on the insole size 45 (without columns Sync and Time)
coordonnees_insole_45 = insoles_coordonnees.loc[0:7, sensor_45]

# Plot : Insoles
if SAVE_CARTOGRAPHY:
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

    plt.savefig("Figures/Cartography_insoles.svg")
    fig.clf()

# Find all the activation
# Load insoles
markers_insole_R = pd.read_csv(
                str(trial_insole_folder) + "/markers_insoles_R_1_L.CSV",
                sep=",",
                decimal=".",
                low_memory=False,
                header=0,
                na_values="NaN",
            )
markers_insole_L = pd.read_csv(
    str(trial_insole_folder) + "/markers_insoles_L_1_R.CSV",
    sep=",",
    decimal=".",
    low_memory=False,
    header=0,
    na_values="NaN",
)

markers_insole_R, markers_insole_L = markers_insole_R.iloc[3:, 1:-1], markers_insole_L.iloc[3:, 1:-1]
markers_insole_R.reset_index(drop=True, inplace=True)
markers_insole_L.reset_index(drop=True, inplace=True)

# Baseline activation
mean_sensor_R_baseline = markers_insole_R.iloc[200:1500, :].mean(axis=0)
mean_sensor_L_baseline = markers_insole_L.iloc[200:1500, :].mean(axis=0)
markers_insole_R_baseline = markers_insole_R.iloc[:, :] - mean_sensor_R_baseline
markers_insole_L_baseline = markers_insole_L.iloc[:, :] - mean_sensor_L_baseline

# Find sensors who were activated
sensor_min_max_R = pd.DataFrame([markers_insole_R_baseline.max(), markers_insole_R_baseline.min()], index=[0, 1], columns=markers_insole_R_baseline.columns.tolist())
sensor_min_max_L = pd.DataFrame([markers_insole_L_baseline.max(), markers_insole_L_baseline.min()], index=[0, 1], columns=markers_insole_L_baseline.columns.tolist())

filtered_sensor_min_max_R = sensor_min_max_R.loc[:, (sensor_min_max_R.iloc[0] > 1.5) & (sensor_min_max_R.iloc[1] > -0.3)]
filtered_sensor_min_max_L = sensor_min_max_L.loc[:, (sensor_min_max_L.iloc[0] > 1.5) & (sensor_min_max_L.iloc[1] >= -0.3)]

# Total activation
markers_insole_R_baseline_total = markers_insole_R_baseline.loc[:, filtered_sensor_min_max_R.columns.to_list()].sum(axis=1)
markers_insole_L_baseline_total = markers_insole_L_baseline.loc[:, filtered_sensor_min_max_L.columns.to_list()].sum(axis=1)

peaks_L, properties_L = signal.find_peaks(markers_insole_L_baseline_total, distance=400, plateau_size=1, height=1.5, width=20)
peaks_R, properties_R = signal.find_peaks(markers_insole_R_baseline_total, distance=300, plateau_size=1, height=1.5, width=20)


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

for i in range(peaks_R.shape[0]):
    # Take 10 frames before and after the peak and find sensors activated
    insole_R_activate_peak = markers_insole_R_baseline.loc[:, filtered_sensor_min_max_R.columns.to_list()][peaks_R[i]-10:peaks_R[i]+10]
       # Sélection des colonnes où les valeurs sont supérieures à zéro
    colonnes_sup_zero_R = insole_R_activate_peak.loc[:, (insole_R_activate_peak > 0).any()]
    coordonnees_R = coordonnees_insole_45.loc[:, colonnes_sup_zero_R.columns.to_list()]
    poids_R = np.mean(colonnes_sup_zero_R, axis=0)
    position_activation_R[0, i] = np.average(coordonnees_R.iloc[6, :], axis=0, weights=poids_R)   #   Position x
    position_activation_R[1, i] = np.average(coordonnees_R.iloc[7, :], axis=0, weights=poids_R) #   Position y

for i in range(peaks_L.shape[0]):
    insole_L_activate_peak = markers_insole_L_baseline.loc[:, filtered_sensor_min_max_L.columns.to_list()][
                             peaks_L[i] - 10:peaks_L[i] + 10]
    colonnes_sup_zero_L = insole_L_activate_peak.loc[:, (insole_L_activate_peak > 0).any()]
    coordonnees_L = coordonnees_insole_45.loc[:, colonnes_sup_zero_L.columns.to_list()]
    poids_L = np.mean(colonnes_sup_zero_L, axis=0)

    position_activation_L[0, i] = np.average(coordonnees_L.iloc[4, :], axis=0, weights=poids_L)   #   Position x
    position_activation_L[1, i] = np.average(coordonnees_L.iloc[5, :], axis=0, weights=poids_L) #   Position y


# Plot position markers semelles
if FLAG_PLOT:
    fig, axs = plt.subplots(2)
    fig.suptitle("Insoles cartography")

        # Subplot 1 : insoles R
    axs[0].plot(coordonnees_insole_45.iloc[7, :], coordonnees_insole_45.iloc[6, :], 'ro')
    axs[0].plot(position_activation_R[1, :], position_activation_R[0, :], 'bo')
    axs[0].set_ylabel("Position Y (mm)", fontsize=14)
    axs[0].title.set_text('Insole Right')

        # Subplot 2 : insoles L
    axs[1].plot(coordonnees_insole_45.iloc[5, :], coordonnees_insole_45.iloc[4, :], 'ro')
    axs[1].plot(position_activation_L[1, :], position_activation_L[0, :], 'bo')
    axs[1].set_xlabel("Position X (mm)", fontsize=14)
    axs[1].set_ylabel("Position Y (mm)", fontsize=14)
    axs[1].title.set_text('Insole Left')
    fig.tight_layout()
    plt.savefig("Figures/insoles_position_markers.svg")
    fig.clf()


