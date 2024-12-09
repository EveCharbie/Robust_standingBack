import os
from pyorerun import BiorbdModel, PhaseRerun
import numpy as np
from save_load_helpers import get_created_data_from_pickle


def search_file(message: str):
    """

    Parameters
    ----------
    message: Message affiché

    Returns
    -------

    """
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window

    currdir = os.getcwd()
    tempdir = filedialog.askopenfilename(parent=root, initialdir=currdir, title=message)
    return tempdir


def visu_pyorerun(name_file_movement: str, name_model: str):
    data = get_created_data_from_pickle(name_file_movement)
    q = data["q_all"]
    t_span = np.vstack(data["time"])
    # pose_propulsion_start = np.array([0, 0, -0.4535, -0.6596, 0.4259, 1.1334, -1.3841, 0.68])
    # q = np.tile(np.arange(len(pose_propulsion_start)), (t_span.shape[0], 1)).T
    # model = biorbd.models(name_model)
    # model.markers(pose_propulsion_start)[0].to_array()
    # model.markers(pose_propulsion_start)[4].to_array()
    # model.markerNames()[0].to_string()

    model = BiorbdModel(name_model)

    viz = PhaseRerun(t_span)
    viz.add_animated_model(model, q)
    viz.rerun("msk_model")
    return viz


# launch in terminal before: rerun --renderer=gl
name_file_movement = search_file("Fichier mouvement")
name_model = search_file("Fichier model")
visu_pyorerun(name_file_movement, name_model)
