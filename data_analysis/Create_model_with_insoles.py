import numpy as np
import biorbd
import bioviz
import ezc3d
import pickle
import pandas as pd
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt
import matplotlib.pyplot as plt


### --- Functions --- ###
def format_vec(vec):
    return ("{} " * len(vec)).format(*vec)[:-1]


DType = TypeVar("DType", bound=np.generic)
Vec3 = Annotated[npt.NDArray[DType], Literal[3]]


class Markers:
    def __init__(self, label: str, parent: str, position: Vec3):
        self.label = label
        self.parent = parent
        self.position = position

    def __str__(self):
        rt = f"\tmarker\t {self.label}\n"
        rt += f"\t\tparent\t{self.parent}\n"
        rt += f"\t\tposition \t{format_vec(self.position)}\n"
        rt += f"\t\ttechnical 1\n"
        rt += f"\tendmarker\n"
        return rt


### --- Parameters --- ###
FLAG_WRITE_BIOMOD = True
FLAG_ANIMATE = False

# load the model
model_path = "EmCo.bioMod"
name_file = "markers_insoles_R_1"
c3d_file = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/29_04_2023/"
path_c3d_reconstructed = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/reconstructions/"

### --- Script --- ###
# Model
model = biorbd.Model(model_path)

# Reconstruct data
with open(str(path_c3d_reconstructed) + str(name_file) + ".pkl", 'rb') as f:
    reconstructed_data = pickle.load(f)

    # Name Dof model / q_recons
Dofs = []
for index in range(model.nbQ()):
    Dofs.append((model.nameDof()[index].to_string(), index))

# Experimental data
c3d = ezc3d.c3d(str(c3d_file) + str(name_file) + ".c3d")
markers = c3d["data"]["points"][:3, :, :] / 1000  # XYZ1 x markers x time_frame
c3d_marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"][:]

markers_insoles_R = []
markers_insoles_L = []
index_insoles_L = []
index_insoles_R = []
marker_model = []

for index, word in enumerate(c3d_marker_labels):
    marker_model.append((word, index))
    if "insole" in word:
        if "G" in word:
            markers_insoles_L.append(word)
            index_insoles_L.append(index)
        if "D" in word:
            markers_insoles_R.append(word)
            index_insoles_R.append(index)

insoles_L = markers[:3, index_insoles_L, :]
insoles_R = markers[:3, index_insoles_R, :]

# Mean position
mean_insole_R = np.nanmean(insoles_R, axis=2)
mean_insole_L = np.nanmean(insoles_L, axis=2)
mean_q_reconstructed = np.nanmean(reconstructed_data['q_recons'], axis=1)

# Difference marqueurs insoles and GlobalJCS (knee_R = 33 ; knee_L = 39)
JCS_knee_R = model.globalJCS(mean_q_reconstructed, 'JambeD').trans().to_array()
JCS_knee_L = model.globalJCS(mean_q_reconstructed, 'JambeG').trans().to_array()
insoles_knee_R = mean_insole_R - JCS_knee_R[:, np.newaxis]
insoles_knee_L = mean_insole_L - JCS_knee_L[:, np.newaxis]
insoles_knee_R[1] *= -1 # Inverse position in y
insoles_knee_L[1] *= -1 # Inverse position in y

# Write on the .BioMod
if FLAG_WRITE_BIOMOD:
    model_insoles = open(str(model_path), "a")

    # Markers JambeD (idx=12)
    model_insoles.write("\n\t\t//Markers\n")
    for i in range(insoles_knee_R.shape[1]):
        model_insoles.write(
            (
                str(
                    Markers(
                        markers_insoles_R[i][8:], model.segments()[12].name().to_string(), insoles_knee_R[:, i]
                    )
                )
            )
        )

    # Markers JambeG (idx=15)
    for i in range(insoles_knee_L.shape[1]):
        model_insoles.write(
            (
                str(
                    Markers(
                        markers_insoles_L[i][8:], model.segments()[15].name().to_string(), insoles_knee_L[:, i]
                    )
                )
            )
        )
    model_insoles.close()

if FLAG_ANIMATE:
    b = bioviz.Viz(loaded_model=model)
    b.exec()