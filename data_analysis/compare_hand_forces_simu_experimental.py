"""
Main code calling scipt python create_cylinder_insole functions to analyze pressure inserts data.
WARNING: the right foot insole was placed on the left tibia and inversely.

forces_control is expressed in the local frame of the insole (side, front), so we keep only the front component, which is the force perpendicular to the tibia in the plane.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import find_peaks


def force_treatment(forces_control, time_control):

    idx_1s = np.where(time_control > 4.5)[0][0]
    idx_6s = np.where(time_control > 7)[0][0]

    # Remove the first 100 samples to remove the base force value
    forces_control_zero = np.nanmean(forces_control[idx_1s:idx_6s, 1])
    force = forces_control[:, 1] - forces_control_zero
    time = time_control

    # Find the maximum force
    idx_max = np.argmax(force)
    begining = idx_max
    end = idx_max
    current_force = force[begining]
    while current_force > 0:
        begining -= 1
        current_force = force[begining]
    current_force = force[end]
    while current_force > 0:
        end += 1
        current_force = force[end]

    begining = 0
    end = -1

    # peaks_idx, _ = find_peaks(force[begining:end], distance=500, prominence=0.8)
    # peaks_values = force[begining:end][peaks_idx]
    # time_peaks = time[begining:end][peaks_idx]


    return force[begining:end], time[begining:end] - time[begining] # , time_peaks - time[begining], peaks_values


fig = plt.figure()

force_results_path = "EmCo_insoles_forces/"
for file in os.listdir(force_results_path):
    if "control" in file:
        with open(f"{force_results_path}/{file}", 'rb') as f:
            data = pickle.load(f)

        forces_control = data["force_data"]
        time_control = data["time"]

        force, time = force_treatment(forces_control, time_control)
        #  peaks_idx, peaks_values

        plt.plot(time, force, label=file)
        # plt.plot(peaks_idx, peaks_values, 'or')

plt.plot([0, 1], [0, 0], '--k')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.savefig("hand_leg_forces_experimental_vs_simulations.png", dpi=300)
plt.show()





