import os

import numpy as np
import pickle

# from plotly.colors import DEFAULT_PLOTLY_COLORS get matlab colors instead
# colors = [color[4:-1].split(",") for color in DEFAULT_PLOTLY_COLORS]
# colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
# colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun
import rerun as rr

folder = "with_noise_same_computer/"
folder_HTC = folder + "HTC"
folder_KTC = folder + "KTC"
folder_FREE = folder + "NTC"

# remove "rgb(" and ")" and split by ","
#  get the data with the smallest cost
time_to_solve = np.zeros((20, 3))
for config, (folder, str_suffix) in enumerate(zip([folder_KTC, folder_HTC, folder_FREE], ["KTC", "HTC", "NTC"])):
    n_files = len([name for name in os.listdir(folder) if name.endswith("_CVG.pkl")])
    for i in range(0, n_files):
        data = pickle.load(open(folder + f"/sol_{i}_CVG.pkl", "rb"))
        time_to_solve[i, config] = data["real_time_to_optimize"]

print(time_to_solve)
print(np.mean(time_to_solve, axis=0))
print(np.std(time_to_solve, axis=0))
# print nicely prettified
print("Mean time to solve")
print(f"KTC: {np.mean(time_to_solve[:, 0])} ± {np.std(time_to_solve[:, 0])} s")
print(f"HTC: {np.mean(time_to_solve[:, 1])} ± {np.std(time_to_solve[:, 1])} s")
print(f"NTC: {np.mean(time_to_solve[:, 2])} ± {np.std(time_to_solve[:, 2])} s")
# in minutes, round to 2 decimals
print("Mean time to solve")
print(f"KTC: {np.mean(time_to_solve[:, 0]) / 60:.2f} ± {np.std(time_to_solve[:, 0]) / 60:.2f} min")
print(f"HTC: {np.mean(time_to_solve[:, 1]) / 60:.2f} ± {np.std(time_to_solve[:, 1]) / 60:.2f} min")
print(f"NTC: {np.mean(time_to_solve[:, 2]) / 60:.2f} ± {np.std(time_to_solve[:, 2]) / 60:.2f} min")
