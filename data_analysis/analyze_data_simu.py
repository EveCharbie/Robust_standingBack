"""
Created on Thu Aug 29 14:31:08 2024

@author: anais
"""

import numpy as np
import pandas as pd
import biorbd
from graph_simu import graph_all_comparaison, get_created_data_from_pickle

# Solution with and without holonomic constraints
path_sol = "/home/mickaelbegon/Documents/Anais/Results_simu"
sol_holo = path_sol + "/" + "Salto_close_loop_landing_5phases_V80.pkl"
sol2 = path_sol + "/" + "Salto_5phases_V11.pkl"
path_model = "/home/mickaelbegon/Documents/Anais/Robust_standingBack/Model/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)

data = pd.read_pickle(sol_holo)
data = get_created_data_from_pickle(sol_holo)
data2 = pd.read_pickle(sol2)

# Difference time
time_diff = data["time"][-1][-1] - data2["time"][-1][-1]

# Duree phase solutions with and without holonomic constraints
time_phase_sol_holo = []
time_phase_sol2 = []

for i in range(len(data["time"])):
    time_phase_sol_holo.append(data["time"][i][-1] - data["time"][i][1])
    time_phase_sol2.append(data2["time"][i][-1] - data2["time"][i][1])

time_diff_phase = [a - b for a, b in zip(time_phase_sol_holo, time_phase_sol2)]

# Diminution des torques
# Par rapport entre premiere et dernier valeur
# Coude
diff_tau_elbow_sol_holo = (data["tau"][2][1][1] - data["tau"][2][1][-1]) / data["tau"][2][1][1] * 100
diff_tau_elbow_sol2 = (data2["tau"][2][1][1] - data2["tau"][2][1][-1]) / data2["tau"][2][1][1] * 100

# Hanche
diff_tau_hip_sol_holo = (data["tau"][2][2][1] - data["tau"][2][2][-1]) / data["tau"][2][2][1] * 100
diff_tau_hip_sol2 = (data2["tau"][2][2][1] - data2["tau"][2][2][-1]) / data2["tau"][2][2][1] * 100

# Par moyenne
# Coude
tau_mean_elbow_sol_holo = np.mean(data["tau"][2][1])
tau_std_elbow_sol_holo = np.std(data["tau"][2][1])
tau_mean_elbow_sol2 = np.mean(data2["tau"][2][1])
tau_std_elbow_sol2 = np.std(data2["tau"][2][1])

# Hanche
tau_mean_hip_sol_holo = np.mean(data["tau"][2][2])
tau_std_hip_sol_holo = np.std(data["tau"][2][2])
tau_mean_hip_sol2 = np.mean(data2["tau"][2][2])
tau_std_hip_sol2 = np.std(data2["tau"][2][2])

# Graphique
graph_all_comparaison(sol_holo, sol2)

# Inertie # TODO: Rajouter boucle

inertie_sol_holo = model.bodyInertia(data["q_all"][:,1]).to_array()[0]
inertie_sol2 = model.bodyInertia(data2["q_all"][:,1]).to_array()[0]

# Energy expenditure (intégrale de la somme de la valeur absolue de tau multiplier par la vitesse angulaire le tout multiplier par dt)