from pyorerun import BiorbdModel, PhaseRerun
import numpy as np
from Save import get_created_data_from_pickle
import tkinter
from tkinter import filedialog
import os

root = tkinter.Tk()
root.withdraw() #use to hide tkinter window


def search_file(message: str):
    """

    Parameters
    ----------
    message: Message affiché

    Returns
    -------

    """
    currdir = os.getcwd()
    tempdir = filedialog.askopenfilename(parent=root, initialdir=currdir, title=message)
    return tempdir


def visu_pyorerun(name_file_movement: str, name_model: str):
    data = get_created_data_from_pickle(name_file_movement)
    q = data["q_all"]
    t_span = np.vstack(data["time"])

    model = BiorbdModel(name_model)

    viz = PhaseRerun(t_span)
    viz.add_animated_model(model, q)
    viz.rerun("msk_model")
    return viz


# launch in terminal before: rerun --renderer=gl
name_file_movement = search_file("Fichier mouvement")
name_model = search_file("Fichier model")
visu_pyorerun(name_file_movement, name_model)
