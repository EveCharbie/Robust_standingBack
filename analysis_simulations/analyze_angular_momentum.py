import numpy as np
import pandas as pd
import biorbd
import pickle

from analysis_simulations.analyze_data_simu import model_adjusted


def adjust_q_with_full_floating_base(q: np.ndarray) -> np.ndarray:
    """
    Adjust the q vector to take into account the full floating base

    Parameters
    ----------
    q: np.ndarray
        The q vector to adjust

    Returns
    -------
    The adjusted q vector
    """
    q_adjusted = np.zeros((q.shape[0] + 3, q.shape[1]))
    q_adjusted[1:4, :] = q[0:3, :]
    q_adjusted[6:, :] = q[3:, :]
    return q_adjusted


def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data_CL.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data_CL: All the data_CL of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp


def plot_vertical_time_lines(time_end_phase_CL, time_end_phase_without, ax, color, linestyle, linewidth):
    color_CL = "tab:orange" if color is None else color
    color_without = "tab:blue" if color is None else color
    linewidth = 0.7 if linewidth is None else linewidth
    ax.axvline(time_end_phase_CL, color=color_CL, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(time_end_phase_without, color=color_without, linestyle=linestyle, linewidth=linewidth)
    return


# print where i am directory
import os

print(os.getcwd())

# Solution with and without holonomic constraints
path_without = "../src/solutions/KTC/"
path_CL = "../src/solutions_CL/HTC/"

# path_model = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"
path_model = "../models/Model2D_7Dof_3C_5M_CL_V3_V3D.bioMod"
model = biorbd.Model(path_model)

CONSIDER_ONLY_CONVERGED = True
if CONSIDER_ONLY_CONVERGED:
    end_file = "CVG.pkl"
else:
    end_file = ".pkl"

min_cost_without = np.inf
for file in os.listdir(path_without):
    if file.endswith(end_file):
        data = pd.read_pickle(path_without + file)
        if data["cost"] < min_cost_without:
            min_cost_without = data["cost"]
            sol_without = path_without + file
print("Min cost without: ", min_cost_without)

min_cost_CL = np.inf
for file in os.listdir(path_CL):
    if file.endswith(end_file):
        data = pd.read_pickle(path_CL + file)
        if data["cost"] < min_cost_CL:
            min_cost_CL = data["cost"]
            sol_CL = path_CL + file
print("Min cost CL: ", min_cost_CL)

data_CL = pd.read_pickle(sol_CL)
data_without = pd.read_pickle(sol_without)

# Angular momentum
ang_mom_CL = np.zeros((data_CL["q_all"].shape[1], 3))
ang_mom_without = np.zeros((data_without["q_all"].shape[1], 3))

adjusted_q = adjust_q_with_full_floating_base(data_without["q_all"])
adjusted_qdot = adjust_q_with_full_floating_base(data_without["qdot_all"])

adjusted_q_CL = adjust_q_with_full_floating_base(data_CL["q_all"])
adjusted_qdot_CL = adjust_q_with_full_floating_base(data_CL["qdot_all"])

for i in range(data_CL["q_all"].shape[1]):
    ang_mom_CL[i, :] = model_adjusted.angularMomentum(adjusted_q[:, i], adjusted_qdot[:, i], True).to_array()
    ang_mom_without[i, :] = model_adjusted.angularMomentum(adjusted_q_CL[:, i], adjusted_qdot_CL[:, i], True).to_array()

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_CL[:, 0],
        mode="lines",
        name="HTC - x",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_without[:, 0],
        mode="lines",
        name="KTC - x",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_CL[:, 1],
        mode="lines",
        name="HTC - y",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_without[:, 1],
        mode="lines",
        name="KTC - y",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_CL[:, 2],
        mode="lines",
        name="HTC - z",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=ang_mom_without[:, 2],
        mode="lines",
        name="KTC - z",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=np.linalg.norm(ang_mom_CL, axis=1),
        mode="lines",
        name="HTC - norm",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=np.linalg.norm(ang_mom_without, axis=1),
        mode="lines",
        name="KTC - norm",
    )
)
fig.show()

comdot_CL = np.zeros((3, data_CL["q_all"].shape[1]))
comdot_without = np.zeros((3, data_without["q_all"].shape[1]))

for i in range(data_CL["q_all"].shape[1]):
    comdot_CL[:, i] = model.CoMdot(adjusted_q_CL[:, i], adjusted_qdot_CL[:, i], True).to_array()
    comdot_without[:, i] = model.CoMdot(adjusted_q[:, i], adjusted_qdot[:, i], True).to_array()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_CL[0, :],
        mode="lines",
        name="HTC - x",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_without[0, :],
        mode="lines",
        name="KTC - x",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_CL[1, :],
        mode="lines",
        name="HTC - y",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_without[1, :],
        mode="lines",
        name="KTC - y",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_CL[2, :],
        mode="lines",
        name="HTC - z",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, data_CL["q_all"].shape[1])),
        y=comdot_without[2, :],
        mode="lines",
        name="KTC - z",
    )
)
fig.show()
