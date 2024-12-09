# Packages
from visualisation import visualisation_movement, graph_all
import os


# Function
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


# Script
name_file_move = search_file("Fichier avec le mouvement")
name_file_model = search_file("Fichier du model")

graph_all(name_file_move)
visualisation_movement(name_file_move, name_file_model)
