import os
import ezc3d
import pandas as pd
from scipy import signal
from pyomeca import Markers, Analogs
import matplotlib.pyplot as plt
import pickle
import numpy as np
import biorbd


### --- Functions --- ###
### --- Synchronisation insoles with motion capture --- ###
# load file (Insoles + MC)

trials_folder_path_insole = "/home/lim/Anais/CollecteStandingBack/EmCo_insoles_rename"
trials_folder_path_MotionCapture = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/29_04_2023"
model_path = "EmCo.bioMod"
model = biorbd.Model(model_path)
Condition = ["salto_control_pre", "salto_control_post"]  #'salto_control_pre',

for i in range(len(Condition)):  # Create a function if everything works
    files_MotionCapture = [
        file
        for file in os.listdir(trials_folder_path_MotionCapture)
        if file.startswith(Condition[i]) and file.endswith(".c3d")
    ]
    files_insole_R = [
        file
        for file in os.listdir(trials_folder_path_insole)
        if file.startswith(Condition[i]) and file.endswith("_R.CSV")
    ]
    files_insole_L = [
        file
        for file in os.listdir(trials_folder_path_insole)
        if file.startswith(Condition[i]) and file.endswith("_L.CSV")
    ]

    if len(files_MotionCapture) == len(files_insole_L) == len(files_insole_R):
        for i in range(len(files_MotionCapture) - 1):
            # Open .c3d
            path_c3d = str(trials_folder_path_MotionCapture) + "/" + files_MotionCapture[i]
            markers = Markers.from_c3d(path_c3d)
            markers = markers[:3, 0:96, :]
            freq_MotionCapture = markers.rate
            analog = Analogs.from_c3d(path_c3d)

            # Open CSV insole
            insole_R = pd.read_csv(
                str(trials_folder_path_insole) + "/" + str(files_MotionCapture[1][:-4]) + "_R.CSV",
                sep=",",
                decimal=".",
                low_memory=False,
                header=0,
                na_values="NaN",
            )
            insole_L = pd.read_csv(
                str(trials_folder_path_insole) + "/" + str(files_MotionCapture[1][:-4]) + "_L.CSV",
                sep=",",
                decimal=".",
                low_memory=False,
                header=0,
                na_values="NaN",
            )
            insole_R, insole_L = insole_R.iloc[3:, :], insole_L.iloc[3:, :]
            insole_R.reset_index(drop=True, inplace=True)
            insole_L.reset_index(drop=True, inplace=True)
            insole_R["Total"] = insole_R.iloc[:, 1:-1].sum(axis=1)
            insole_L["Total"] = insole_L.iloc[:, 1:-1].sum(axis=1)
            freq_insoles = round(insole_R.shape[0] / float(insole_R.iloc[-1, 0]))

            # Find the pics after begin close loop constraints
            # Insoles
            peaks_total, _ = signal.find_peaks(insole_R["Total"], height=90)
            mins_total, _ = signal.find_peaks(insole_R["Total"] * -1, height=-25, distance=200)
            begin_movement_insole = max([x for x in mins_total if x < peaks_total[0]])
            peaks_sync, _ = signal.find_peaks(insole_R["Sync"], height=900)
            first_peak_movement_insole = min([x for x in peaks_sync if x > begin_movement_insole])

            # Motion capture (Find n°frame of peak ==> distance marker hand and genou < value and stable)
            # Calulate middle hand (MIDMETAC3D markers.loc['x', 'MIDMETAC3D'], METAC2D markers[0][37], METAC5D markers[0][38]) [en x et y]
            middle_hand = (
                abs(
                    markers.loc[["x", "y"], "MIDMETAC3D"]
                    + markers.loc[["x", "y"], "METAC2D"]
                    + markers.loc[["x", "y"], "METAC5D"]
                )
            ) / 3

            # Calculate middle knee (CONDEXTD markers[0][53], CONDINTD markers[0][54]) [ en x et y]
            middle_knee = (abs(markers.loc[["x", "y"], "CONDINTD"] + markers.loc[["x", "y"], "CONDEXTD"])) / 2

            # Calculate distance between middle hand and knee
            distance_knee_hand = np.sqrt(
                ((middle_hand.loc["x"] - middle_knee.loc["x"]) * (middle_hand.loc["x"] - middle_knee.loc["x"]))
                + ((middle_hand.loc["y"] - middle_knee.loc["y"]) * (middle_hand.loc["y"] - middle_knee.loc["y"]))
            )
            # Find first minima
            begin_movement_MotionCapture = np.argmin(distance_knee_hand.to_numpy())
            peaks_sync_MotionCapture, _ = signal.find_peaks(analog.loc["Time.1"], height=1)
            first_peak_movement_MotionCapture = min(
                [x for x in peaks_sync_MotionCapture if x > begin_movement_MotionCapture]
            )

            # Cut data to be synchronize
            # Cut the end
            peak_to_last_markers = markers.shape[2] - first_peak_movement_MotionCapture
            peak_to_last_insoles = round(
                (insole_R.shape[0] - first_peak_movement_insole) / 2 - 0.8
            )  # Peak for a frequency at 200Hz
            diff_peak_to_last = abs(peak_to_last_markers - peak_to_last_insoles)
            diff_peak = abs(first_peak_movement_MotionCapture - round(first_peak_movement_insole / 2 - 0.8))

            if peak_to_last_insoles > peak_to_last_markers:
                insole_R = insole_R.loc[0 : (insole_R.shape[0] - diff_peak_to_last * 2), :]
                insole_L = insole_L.loc[0 : (insole_L.shape[0] - diff_peak_to_last * 2), :]
            elif peak_to_last_insoles < peak_to_last_markers:
                distance_knee_hand = distance_knee_hand.to_numpy()[0 : (analog.shape[1] - diff_peak_to_last)]
                analog = analog.to_numpy()[:, 0 : (analog.shape[1] - diff_peak_to_last)]
                markers = markers.to_numpy()[:, :, 0 : (markers.shape[2] - diff_peak_to_last)]
            else:
                pass

                # Cut the beginning
            if first_peak_movement_insole / 2 > first_peak_movement_MotionCapture:
                insole_R = insole_R.loc[diff_peak * 2 :, :]
                insole_L = insole_L.loc[diff_peak * 2 :, :]
            elif first_peak_movement_insole / 2 < first_peak_movement_MotionCapture:
                distance_knee_hand = distance_knee_hand.to_numpy()[diff_peak:]
                analog = analog.to_numpy()[:, diff_peak:]
                markers = markers.to_numpy()[:, :, diff_peak:]
            else:
                pass

            # Visualisation
            # Creation vecteur temps
            insole_R["Time"] = np.arange(0, insole_R.shape[0] * 1 / 400, 1 / 400)
            vector_time_marker = np.arange(0, distance_knee_hand.shape[0] * 1 / 200, 1 / 200)
            # Plot
            plt.plot(insole_R["Time"], insole_R["Total"], "o", label="Total_pression_insoles")
            plt.plot(vector_time_marker, distance_knee_hand, "s", label="Distance_knee_main")
            plt.legend()
            plt.show()

        # Return file sync
    else:
        print("The number of file for the motion capture and the insoles is not the same")

### --- Integration markers to model --- ###

### --- Represent insoles by a cylinder --- ###

### --- Vectoriel sum of cell of insoles --- ###
