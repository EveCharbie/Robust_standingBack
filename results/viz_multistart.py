import numpy as np
import pickle
from plotly.colors import DEFAULT_PLOTLY_COLORS
from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun

folder = "/home/pierre/Projets_Python/Robust_standingBack/sandbox/best_solutions/with_noise/HTC"
model_path = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"

# remove "rgb(" and ")" and split by ","
colors = [color[4:-1].split(",") for color in DEFAULT_PLOTLY_COLORS]
colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

# Charger les données
phase_reruns = []
for i in range(0, 19):
    data = pickle.load(open(folder + f"/sol_{i}_CVG.pkl", "rb"))
    time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
    q = np.concatenate([np.array(q) for q in data["q"]], axis=1)

    phase_reruns.append(PhaseRerun(t_span=time, window=f"simulation_{i}"))
    m = BiorbdModel(model_path)
    m.options.mesh_color = colors[i - 1]
    phase_reruns[-1].add_animated_model(m, q)
    phase_reruns[-1].rerun()

mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
mrr2.rerun("yo")
