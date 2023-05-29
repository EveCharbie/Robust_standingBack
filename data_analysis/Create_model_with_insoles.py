import numpy as np
import biorbd
import bioviz
import ezc3d
import pandas as pd
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt
import matplotlib.pyplot as plt


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
        rt = f"marker\t {self.label}\n"
        rt += f"\tparent\t{self.parent}\n"
        rt += f"\tposition {format_vec(self.position)}\n"
        rt += f"\ttechnical 1\n"
        rt += f"endmarker\n"
        return rt


# --------------------------------------------------------------

FLAG_ANIMATE = True
FLAG_CREATE = False
# load the model
# model_path = "EmCo.bioMod"
model_path = "Model_insoles.bioMod"
model = biorbd.Model(model_path)

c3d_file = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/29_04_2023/markers_insoles_R_1.c3d"
# c3d_file = "/home/lim/Anais/CollecteStandingBack/EmCo_motion_capture/EmCo/29_04_2023/salto_control_post_2.c3d"
c3d = ezc3d.c3d(c3d_file)
markers = c3d["data"]["points"][:3, :, :] / 1000  # XYZ1 x markers x time_frame
c3d_marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"][:]

if FLAG_CREATE:
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
    plt.plot(insoles_L[2, 10, :])
    plt.legend()
    plt.show()

    # Add markers insoles to the leg segment (12 = JambeD, 15 = JambeG)
    mean_insole_R = np.nanmean(insoles_R, axis=2)
    mean_insole_L = np.nanmean(insoles_L, axis=2)
    mean_markers = np.nanmean(markers, axis=2)

    # # create numpy array difference distance CONTEXTG (97) and CONTEXTD (80) and insoles markers
    diff_knee_insole_R = pd.DataFrame(
        0,
        index=np.arange(insoles_R.shape[1]),
        columns=["x", "y", "z"],
    )
    row_name_mapping_R = {
        old_name: new_name[8:] for old_name, new_name in zip(diff_knee_insole_R.index, markers_insoles_R)
    }  # Create a dictionary mapping old row names to new row names
    diff_knee_insole_R = diff_knee_insole_R.rename(index=row_name_mapping_R)

    diff_knee_insole_L = pd.DataFrame(
        0,
        index=np.arange(insoles_L.shape[1]),
        columns=["x", "y", "z"],
    )
    row_name_mapping_L = {
        old_name: new_name[8:] for old_name, new_name in zip(diff_knee_insole_L.index, markers_insoles_L)
    }  # Create a dictionary mapping old row names to new row names
    diff_knee_insole_L = diff_knee_insole_L.rename(index=row_name_mapping_L)

    for j in range(mean_insole_R.shape[1]):
        diff_knee_insole_R.iloc[j] = mean_insole_R[:, j] - mean_markers[:, 80]
        diff_knee_insole_L.iloc[j] = mean_insole_L[:, j] - mean_markers[:, 97]

    # Distance between CONTEXTG(97) and CONTEXTD (80) and the global reference

    name_old_markers = []
    for index in range(model.nbMarkers()):
        name_old_markers.append((model.markerNames()[index].to_string(), index))

    pos_CONTEXTD = model.markers()[63].to_array()
    pos_CONTEXTG = model.markers()[80].to_array()

    pos_insole_R = diff_knee_insole_R
    pos_insole_L = diff_knee_insole_L

    for j in range(diff_knee_insole_R.shape[1]):
        pos_insole_R.iloc[j] = diff_knee_insole_R.iloc[j, :] - pos_CONTEXTD
        pos_insole_L.iloc[j] = diff_knee_insole_L.iloc[j, :] - pos_CONTEXTG

    # # Write on the .BioMod
    model_insoles = open("Model_insoles.bioMod", "w")

    # Markers JambeD (idx=12)
    model_insoles.write("\t//Markers\n")
    for i in range(insoles_R.shape[1]):
        model_insoles.write(
            (
                str(
                    Markers(
                        pos_insole_R.index[i], model.segments()[12].name().to_string(), pos_insole_R.iloc[i].to_numpy()
                    )
                )
            )
        )

    # Markers JambeG (idx=15)
    model_insoles.write("\t//Markers\n")
    for i in range(insoles_L.shape[1]):
        model_insoles.write(
            (
                str(
                    Markers(
                        pos_insole_L.index[i], model.segments()[15].name().to_string(), pos_insole_L.iloc[i].to_numpy()
                    )
                )
            )
        )
    model_insoles.close()


if FLAG_ANIMATE:
    b = bioviz.Viz(loaded_model=model)
    b.load_experimental_markers(data=markers)
    b.exec()
