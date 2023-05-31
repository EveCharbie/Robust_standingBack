import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

### --- Functions --- ###

def lissage(signal_brut,L):
    res = np.copy(signal_brut)  # duplication des valeurs
    for i in range(1, len(signal_brut) - 1):  # toutes les valeurs sauf la première et la dernière
        L_g = min(i, L)  # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut)-i-1,L) # nombre de valeurs disponibles à droite
        Li = min(L_g, L_d)
        res[i] = np.sum(signal_brut[i - Li:i + Li + 1]) / (2 * Li + 1)
    return res

### --- Parameters --- ###

SAVE_CARTOGRAPHY = False
FLAG_ACTIVATION = True

### --- Cartography insoles --- ###
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

markers_insole_R, markers_insole_L = markers_insole_R.iloc[3:, :-1], markers_insole_L.iloc[3:, :-1]
markers_insole_R.reset_index(drop=True, inplace=True)
markers_insole_L.reset_index(drop=True, inplace=True)

# Baseline activation
# mean_sensor_R_baseline = markers_insole_R.iloc[:1000, 1:].sum(axis=0)
# mean_sensor_L_baseline = markers_insole_L.iloc[:1000, 1:].sum(axis=0)
# mean_sensor_R = markers_insole_R.iloc[:, 1:].sum(axis=0)
# mean_sensor_L = markers_insole_L.iloc[:, 1:].sum(axis=0)
# markers_insole_R_baseline = markers_insole_R.iloc[:, 1:] - mean_sensor_R_baseline
# markers_insole_L_baseline = markers_insole_L.iloc[:, 1:] - mean_sensor_L_baseline
# markers_insole_R = markers_insole_R.iloc[:, 1:] - mean_sensor_R
# markers_insole_L = markers_insole_L.iloc[:, 1:] - mean_sensor_L

# Total activation
markers_insole_R["Total"] = markers_insole_R.iloc[:, 1:].sum(axis=1)
markers_insole_L["Total"] = markers_insole_L.iloc[:, 1:].sum(axis=1)

# Lissage du signal pour trouver les pics
signal_lisse_R = lissage(markers_insole_R["Total"], 50)
signal_lisse_L = lissage(markers_insole_L["Total"], 50)

# Plot
# plt.title("Difference signal buités et avec lissage insole right")
# plt.plot(signal_lisse_R, color="red", label="signal lissé")
# plt.plot(markers_insole_R["Total"], color="blue", label="signal bruité")
# plt.legend()
# plt.show()

# plt.title("Difference signal buités et avec lissage insole left")
# plt.plot(signal_lisse_L, color="red", label="signal lissé")
# plt.plot(markers_insole_L["Total"], color="blue", label="signal bruité")
# plt.legend()
# plt.show()


# markers_insole_R_baseline["Total"] = markers_insole_R_baseline.iloc[:, 1:].sum(axis=1)
# markers_insole_L_baseline["Total"] = markers_insole_L_baseline.iloc[:, 1:].sum(axis=1)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
# fig.suptitle(" Totale activation insole rightwith and without baseline")
# axes[0].plot(markers_insole_R["Total"], color="blue")
# axes[0].set_ylabel("Pressure (N/cm2)", fontsize=14)
# axes[0].set_xlabel("Time", fontsize=14)
# axes[1].plot(markers_insole_R_baseline["Total"], color="green")
# axes[1].set_ylabel("Pressure (N/cm2)", fontsize=14)
# axes[1].set_xlabel("Time", fontsize=14)
# fig.tight_layout()
# plt.show()

peaks, peak_plateau = signal.find_peaks(-signal_lisse_R, distance=300, plateau_size=1)
plt.plot(signal_lisse_R)
plt.plot(peaks, signal_lisse_R[peaks], "x")
plt.show()

if FLAG_ACTIVATION:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    fig.suptitle(" Totale activation insole right and left")
    axes[0].plot(markers_insole_R["Total"], color="blue")
    axes[0].set_ylabel("Pressure (N/cm2)", fontsize=14)
    axes[0].set_xlabel("Time", fontsize=14)
    axes[1].plot(markers_insole_L["Total"], color="green")
    axes[1].set_xlabel("Time", fontsize=14)
    fig.tight_layout()
    plt.savefig("Figures/insoles_totale_activation_markers.svg")
    fig.clf()
