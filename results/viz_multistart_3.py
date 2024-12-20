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

# get matplotlib colors instead
colors = [
    # tab:blue
    (31, 119, 180),
    # tab:orange
    (255, 127, 14),
    # tab:green
    (44, 160, 44),
    # tab:red
    (214, 39, 40),
    # tab:purple
    (148, 103, 189),
    # tab:brown
    (140, 86, 75),
    # tab:pink
    (227, 119, 194),
    # tab:gray
    (127, 127, 127),
    # tab:olive
    (188, 189, 34),
    # tab:cyan
    (23, 190, 207),
]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

folder = "with_noise/"
folder_HTC = folder + "HTC"
folder_KTC = folder + "KTC"
folder_FREE = folder + "NTC"
model_path = "../models/Model2D_7Dof_3C_5M_CL_V3_less_markers.bioMod"

# remove "rgb(" and ")" and split by ","
#  get the data with the smallest cost
file_idx = []
for config, (folder, str_suffix) in enumerate(zip([folder_KTC, folder_HTC, folder_FREE], ["KTC", "HTC", "NTC"])):
    n_files = len([name for name in os.listdir(folder) if name.endswith("_CVG.pkl")])
    smallest_idx, smallest_value = 0, np.inf
    for i in range(0, n_files):
        data = pickle.load(open(folder + f"/sol_{i}_CVG.pkl", "rb"))
        if data["cost"] < smallest_value:
            smallest_value = data["cost"]
            smallest_idx = i
            print(f"New smallest value of {str_suffix} : {smallest_value} at index {smallest_idx}")
    file_idx.append(smallest_idx)


# Charger les données
for config, (folder, str_suffix, file_id) in enumerate(
    zip([folder_KTC, folder_HTC, folder_FREE], ["KTC", "HTC", "NTC"], file_idx)
):
    #  get the number of _CVG.pkl files in the folder
    phase_reruns = []
    n_files = len([name for name in os.listdir(folder) if name.endswith("_CVG.pkl")])
    data = pickle.load(open(folder + f"/sol_{file_id}_CVG.pkl", "rb"))
    time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
    q = np.concatenate([np.array(q) for q in data["q"]], axis=1)

    phase_reruns.append(PhaseRerun(t_span=time[:-1], window=f"s_{file_id}_{str_suffix}"))
    m = BiorbdModel(model_path)
    m.options.mesh_color = colors[config]
    phase_reruns[-1].add_animated_model(m, q[:, :-1])
    # phase_reruns[-1].add_force_data(data["force"])

    mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
    mrr2.rerun(str_suffix)

    rr.log(
        "world/cam",
        rr.Pinhole(
            fov_y=0.7853982,
            aspect_ratio=1.7777778,
            camera_xyz=rr.ViewCoordinates.FLU,
            image_plane_distance=3,
        ),
    )
    rr.log(
        "world/cam",
        rr.Transform3D(
            translation=[-3, 0, 1.05],
        ),
    )
