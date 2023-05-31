import os
import pandas as pd
from scipy import signal
from pyomeca import Markers, Analogs
import matplotlib.pyplot as plt
import numpy as np
import biorbd
import pickle

### --- Functions --- ###

### --- Parameters --- ###

trials_folder_path_insole = "/home/lim/Anais/CollecteStandingBack/EmCo_insoles_rename"
trials_folder_path_MotionCapture = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/29_04_2023"
path_folder_save = "/home/lim/Anais/CollecteStandingBack"
model_path = "EmCo.bioMod"
model = biorbd.Model(model_path)
Condition = [
    # "salto_control_pre",
    "salto_control_post",
]
FLAG_SAVE = True
FLAG_GRAPH = True

### --- Synchronisation insoles with motion capture --- ###


#---------------

for number_condition in range(len(Condition)):  # Create a function if everything works
    files_MotionCapture = [
        file
        for file in os.listdir(trials_folder_path_MotionCapture)
        if file.startswith(Condition[number_condition]) and file.endswith(".c3d")
    ]
    files_insole_R = [
        file
        for file in os.listdir(trials_folder_path_insole)
        if file.startswith(Condition[number_condition]) and file.endswith("_R.CSV")
    ]
    files_insole_L = [
        file
        for file in os.listdir(trials_folder_path_insole)
        if file.startswith(Condition[number_condition]) and file.endswith("_L.CSV")
    ]

    if len(files_MotionCapture) == len(files_insole_L) == len(files_insole_R):
        for i in range(len(files_MotionCapture)):
            # Open .c3d
            path_c3d = str(trials_folder_path_MotionCapture) + "/" + files_MotionCapture[i]
            markers = Markers.from_c3d(path_c3d)
            markers = markers[:3, 0:96, :]
            freq_MotionCapture = markers.rate
            analog = Analogs.from_c3d(path_c3d)

            # Open CSV insole
            insole_R = pd.read_csv(
                str(trials_folder_path_insole) + "/" + str(files_MotionCapture[i][:-4]) + "_R.CSV",
                sep=",",
                decimal=".",
                low_memory=False,
                header=0,
                na_values="NaN",
            )
            insole_L = pd.read_csv(
                str(trials_folder_path_insole) + "/" + str(files_MotionCapture[i][:-4]) + "_L.CSV",
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

            peaks_total, _ = signal.find_peaks(insole_R["Total"], height=90, distance=50)
            # mins_total, _ = signal.find_peaks(insole_R["Total"] * -1, height=-25, distance=200)
            begin_movement_insole = peaks_total[0]
            peaks_sync, _ = signal.find_peaks(insole_R["Sync"], height=900)
            first_peak_movement_insole = min([x for x in peaks_sync if x > peaks_total[0]])

            # Motion capture (Find n°frame of peak ==> distance marker hand and genou < value and stable)
            # Calulate middle hand (MIDMETAC3D markers.loc['x', 'MIDMETAC3D'], METAC2D markers[0][37], METAC5D markers[0][38]) [en x et y]
            middle_hand_L = (
                abs(
                    markers.loc[["x", "y"], "MIDMETAC3G"]
                    + markers.loc[["x", "y"], "METAC2G"]
                    + markers.loc[["x", "y"], "METAC5G"]
                )
            ) / 3

            # Calculate middle knee (CONDEXTD markers[0][53], CONDINTD markers[0][54]) [ en x et y]
            middle_knee_L = (abs(markers.loc[["x", "y"], "CONDINTG"] + markers.loc[["x", "y"], "CONDEXTG"])) / 2

            # Calculate distance between middle hand and knee
            distance_knee_hand_L = np.sqrt(
                ((middle_hand_L.loc["x"] - middle_knee_L.loc["x"]) * (middle_hand_L.loc["x"] - middle_knee_L.loc["x"]))
                + (
                    (middle_hand_L.loc["y"] - middle_knee_L.loc["y"])
                    * (middle_hand_L.loc["y"] - middle_knee_L.loc["y"])
                )
            )

            # Find first minima
            begin_movement_MotionCapture = np.nanargmin(distance_knee_hand_L.to_numpy())
            peaks_total_motion_capture, _ = signal.find_peaks(distance_knee_hand_L, height=500)
            peaks_sync_MotionCapture, _ = signal.find_peaks(
                analog.loc["Time.1"][peaks_total_motion_capture[1]:], height=1
            )
            peaks_sync_MotionCapture = peaks_sync_MotionCapture + peaks_total_motion_capture[1]
            first_peak_movement_MotionCapture = min(
                [x for x in peaks_sync_MotionCapture if x > begin_movement_MotionCapture]
            )

            # Cut data to be synchronize
            # Cut the end
            peak_to_last_markers = markers.shape[2] - first_peak_movement_MotionCapture
            peak_to_last_insoles = round(
                (insole_R.shape[0] - first_peak_movement_insole) / 2
            )  # Peak for a frequency at 200Hz
            diff_peak_to_last = abs(peak_to_last_markers - peak_to_last_insoles)
            diff_peak = abs(first_peak_movement_MotionCapture - round(first_peak_movement_insole / 2))

            if peak_to_last_insoles > peak_to_last_markers:
                # if ((insole_R.shape[0] - first_peak_movement_insole) / 2) % 2 == 0:
                insole_R = insole_R.loc[0: (insole_R.shape[0] - diff_peak_to_last * 2), :]
                insole_L = insole_L.loc[0: (insole_L.shape[0] - diff_peak_to_last * 2), :]
                # else:
                #     insole_R = insole_R.loc[0: (insole_R.shape[0] - diff_peak_to_last * 2) - 1, :]
                #     insole_L = insole_L.loc[0: (insole_L.shape[0] - diff_peak_to_last * 2) - 1, :]
            elif peak_to_last_insoles < peak_to_last_markers:
                if ((insole_R.shape[0] - first_peak_movement_insole) / 2) % 2 == 0:
                    distance_knee_hand_L = distance_knee_hand_L[0: (analog.shape[1] - diff_peak_to_last)].to_numpy()
                    analog = analog[:, 0: (analog.shape[1] - diff_peak_to_last)].to_numpy()
                    markers = markers[:, :, 0: (markers.shape[2] - diff_peak_to_last)].to_numpy()
                else:
                    distance_knee_hand_L = distance_knee_hand_L[
                        0: (analog.shape[1] - (diff_peak_to_last + 1))
                    ].to_numpy()
                    analog = analog[:, 0: (analog.shape[1] - (diff_peak_to_last + 1))].to_numpy()
                    markers = markers[:, :, 0: (markers.shape[2] - (diff_peak_to_last + 1))].to_numpy()
                    insole_R = insole_R.loc[0: insole_R.shape[0] - 2, :]
                    insole_L = insole_L.loc[0: insole_L.shape[0] - 2, :]
            else:
                pass

                # Cut the beginning
            if first_peak_movement_insole / 2 > first_peak_movement_MotionCapture:
                insole_R = insole_R.loc[diff_peak * 2:, :]
                insole_L = insole_L.loc[diff_peak * 2:, :]
                # begin_movement_insole = begin_movement_insole - (diff_peak * 2)
                # first_peak_movement_insole = first_peak_movement_insole - (diff_peak * 2)
            elif first_peak_movement_insole / 2 < first_peak_movement_MotionCapture:    # Boucle pas bon
                if (first_peak_movement_insole / 2) % 2 == 0:
                    distance_knee_hand_L = distance_knee_hand_L[diff_peak:]
                    analog = analog[:, diff_peak:]
                    markers = markers[:, :, diff_peak:]
                    begin_movement_MotionCapture = begin_movement_MotionCapture - diff_peak
                    first_peak_movement_MotionCapture = first_peak_movement_MotionCapture - diff_peak
                else:
                    distance_knee_hand_L = distance_knee_hand_L[diff_peak + 1:]
                    analog = analog[:, diff_peak + 1:]
                    markers = markers[:, :, diff_peak + 1:]
                    insole_R = insole_R.loc[2:, :]
                    insole_L = insole_L.loc[2:, :]
                    begin_movement_MotionCapture = begin_movement_MotionCapture - diff_peak
                    first_peak_movement_MotionCapture = first_peak_movement_MotionCapture - diff_peak
            else:
                pass

            # Visualisation and save fig
            # Creation vecteur temps
            insole_R["Time"] = np.arange(0, insole_R.shape[0] * 1 / 400, 1 / 400)
            insole_L["Time"] = np.arange(0, insole_L.shape[0] * 1 / 400, 1 / 400)
            vector_time_marker = np.arange(0, distance_knee_hand_L.shape[0] * 1 / 200, 1 / 200)

            # Plot 1: Peak insole (subplot1) and peak motion capture (subplot2)
            if FLAG_GRAPH:
                fig, axs = plt.subplots(2)
                fig.suptitle(" Comparison beginning of movement and peak sync of motion capture and insoles")

                # Make plot 1 (Insoles)
                axs[0].plot(
                    insole_R["Time"], insole_R["Total"], color="green", linestyle="solid", label="Total_pression_insoles_R"
                )
                axs[0].plot(
                    insole_R["Time"], insole_L["Total"], color="blue", linestyle="solid", label="Total_pression_insoles_L"
                )
                axs[0].plot(
                    insole_R["Time"][begin_movement_insole],
                    insole_R["Total"][begin_movement_insole],
                    "go",
                    label="beginning movement insole",
                )
                axs[0].set_ylabel("Pressure (N/cm2)", fontsize=14, color="blue")
                axs[0].legend(loc="upper left")
                axs[0].axvline(x=insole_R["Time"][first_peak_movement_insole], color="r", label="Peak sync insole")

                # Make plot 2 (Markers)
                axs[1].plot(
                    vector_time_marker,
                    distance_knee_hand_L,
                    color="orange",
                    linestyle="solid",
                    label="Distance_knee_main_L",
                )
                axs[1].plot(
                    vector_time_marker[begin_movement_MotionCapture],
                    distance_knee_hand_L[begin_movement_MotionCapture],
                    "yo",
                    label="Beginning movement motion capture",
                )
                axs[1].set_ylabel("Distance (in mm)", fontsize=14, color="orange")
                axs[1].legend(loc="upper left")
                axs[1].set_xlabel("Time (s)", fontsize=14)
                axs[1].axvline(
                    x=vector_time_marker[first_peak_movement_MotionCapture], color="r", label="Peak sync motion capture"
                )
                plt.savefig("Figures/Sync_insoles_MC_" + str(files_MotionCapture[i][:-4]) + ".svg")
                fig.clf()

                print(
                    "Nom fichier:" + str(files_MotionCapture[i]),
                    "Markers shape:" + str(markers.shape),
                    "Analog shape:" + str(analog.shape),
                    "Insole_R shape:" + str(insole_R.shape),
                    "Insole_L shape:" + str(insole_L.shape),
                    "Diff:" + str(insole_R.shape[0] / 2 - markers.shape[2]),
                )

            # Return file sync
            if FLAG_SAVE:
                save_path = str(path_folder_save) + "/reconstructions/" + str(files_MotionCapture[i][:-4]) + ".pkl"
                with open(save_path, "wb") as f:
                    data = {
                        "insole_R": insole_R,
                        "insole_L": insole_L,
                        "motion_capture_markers": markers,
                        "motion_capture_analog": analog,

                    }
                    pickle.dump(data, f)

    else:
        print("The number of file for the motion capture and the insoles is not the same")

### --- Integration markers to model --- ###

### --- Represent insoles by a cylinder --- ###

### --- Vectoriel sum of cell of insoles --- ###
