import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import product

# Charger les données

from pyorerun import PhaseRerun, BiorbdModel, MultiPhaseRerun

model = BiorbdModel("../../models/Model2D_7Dof_2C_5M_CL_V3.bioMod")

data = pickle.load(
    open(
        "/home/pierre/Projets_Python/Robust_standingBack/holonomic_research/solutions/Salto_5phases_VPierre_taudot1/sol_0_CVG.pkl",
        "rb",
    )
)

# Extraire les contraintes
constraints = np.array(data["constraints"]).squeeze()

# Bar chart des contraintes
fig = px.bar(x=np.arange(len(constraints)), y=constraints)
fig.show()


# Fonction pour étendre les bornes
def extend_bounds(data_var, bounds_var, prefix):
    bounds_var_extended = []
    for sub_var, bound_var in zip(data[data_var], bounds_var):
        bound_var_extended = np.zeros(sub_var.shape)
        try:
            bound_var_extended[:, 0] = bound_var[:, 0]
            bound_var_extended[:, 1:-1] = bound_var[:, 1:2]
            bound_var_extended[:, -1] = bound_var[:, -1]
        except:
            bound_var_extended[:, :] = np.nan
        bounds_var_extended.append(bound_var_extended)
    return np.concatenate(bounds_var_extended, axis=1)


# Fonction pour créer les sous-graphiques
def create_subplots(data_var, min_bounds, max_bounds, var_name, vertical_idx_offset=0):
    # Créer une figure de sous-graphiques avec 2 lignes et 4 colonnes
    fig = make_subplots(rows=2, cols=4, shared_xaxes=True, vertical_spacing=0.02)

    # Ajouter chaque trace aux sous-graphiques correspondants
    for row, (i, j) in enumerate(product(range(1, 3), range(1, 5))):
        if row < vertical_idx_offset:
            continue
        row -= vertical_idx_offset
        fig.add_trace(
            go.Scatter(x=np.arange(data_var.shape[1]), y=data_var[row, :], mode="markers", name=f"{var_name}{i}"),
            row=i,
            col=j,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(min_bounds.shape[1]),
                y=min_bounds[row, :],
                mode="lines+markers",
                name=f"min_bound_{var_name}{i}",
                marker=dict(color="black", size=3),
                line=dict(color="black", width=1),
            ),
            row=i,
            col=j,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(max_bounds.shape[1]),
                y=max_bounds[row, :],
                mode="lines+markers",
                name=f"max_bound_{var_name}{i}",
                marker=dict(color="black", size=3),
                line=dict(color="black", width=1),
            ),
            row=i,
            col=j,
        )

    # Mettre à jour la disposition
    fig.update_layout(height=1000, title_text=f"Subplots of {var_name} values")

    # Afficher la figure
    fig.show()


# Traitement pour q
q = np.concatenate([np.array(q) for q in data["q"]], axis=1)
min_bounds_q = extend_bounds("q", data["min_bounds_q"], "q")
max_bounds_q = extend_bounds("q", data["max_bounds_q"], "q")

# Traitement pour qdot
qdot = np.concatenate([np.array(qdot) for qdot in data["qdot"]], axis=1)
min_bounds_qdot = extend_bounds("qdot", data["min_bounds_qdot"], "qdot")
max_bounds_qdot = extend_bounds("qdot", data["max_bounds_qdot"], "qdot")

# Traitement pour tau
tau = np.concatenate([np.array(tau) for tau in data["tau"]], axis=1)
min_bounds_tau = extend_bounds("tau", data["min_bounds_tau"], "tau")
max_bounds_tau = extend_bounds("tau", data["max_bounds_tau"], "tau")

# Traitement pour taudot
taudot = np.concatenate([np.array(taudot) for taudot in data["taudot"]], axis=1)
min_bounds_taudot = extend_bounds("taudot", data["min_bounds_taudot"], "taudot")
max_bounds_taudot = extend_bounds("taudot", data["max_bounds_taudot"], "taudot")

# Créer les sous-graphiques pour q, qdot, et tau
create_subplots(q, min_bounds_q, max_bounds_q, "q")
create_subplots(qdot, min_bounds_qdot, max_bounds_qdot, "qdot")
create_subplots(tau, min_bounds_tau, max_bounds_tau, "tau", vertical_idx_offset=3)
create_subplots(taudot, min_bounds_taudot, max_bounds_taudot, "taudot", vertical_idx_offset=3)
