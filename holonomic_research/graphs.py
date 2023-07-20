import numpy as np
import pickle
from Save import get_created_data_from_pickle
import matplotlib.pyplot as plt
from bioptim import BiorbdModel
from biorbd import segment_index

def constraints_graphs(ocp, sol):
    """Plot the graphs of the constraints"""

    simple_derivative0 = []
    simple_derivative1 = []
    for qi, qdoti in zip(sol.states["u"].T, sol.states["udot"].T):
        print(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti))
        simple_derivative0.append(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti)[0].toarray()[0, 0])
        simple_derivative = np.array(simple_derivative0).squeeze()
        simple_derivative1.append(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti)[1].toarray()[0, 0])
        simple_derivative = np.array(simple_derivative1).squeeze()

    double_derivative_0 = []
    double_derivative_1 = []
    # print(sol.controls["tau"])
    for qi, qdoti, taui in zip(sol.states["u"].T, sol.states["udot"].T, sol.controls["tau"].T):
        qddoti = ocp.nlp[0].dynamics_func(np.hstack((qi, qdoti)), taui, 0)[ocp.nlp[0].model.nb_qdot :]
        # print(ocp.nlp[0].model.holonomic_constraints_double_derivative(qi, qdoti, qddoti))
        double_derivative_0.append(
            ocp.nlp[0].model.holonomic_constraints_double_derivative(qi, qdoti, qddoti)[0].toarray()[0, 0]
        )
        double_derivative = np.array(double_derivative_0).squeeze()
        double_derivative_1.append(
            ocp.nlp[0].model.holonomic_constraints_double_derivative(qi, qdoti, qddoti)[1].toarray()[0, 0]
        )
        double_derivative = np.array(double_derivative_1).squeeze()

    # --- Show results --- #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1)
    # constraints
    ax[0].plot(sol.time, ocp.nlp[0].model._holonomic_constraints[0](sol.states["u"]).T)
    ax[0].set_title("Holonomic constraints (position)")
    #
    ax[1].set_title("Holonomic constraints derivative (velocity)")
    ax[1].plot(sol.time, simple_derivative0, label="d0")
    ax[1].plot(sol.time, simple_derivative1, label="d1")
    #
    ax[2].set_title("Holonomic constraints double derivative (acceleration)")
    ax[2].plot(sol.time, double_derivative_0, label="dd0")
    ax[2].plot(sol.time, double_derivative_1, label="dd1")
    plt.show()

def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp

# Info
pickle_1 = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V13.pkl"
pickle_2 = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V10.pkl"
name_file_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_7Dof_0C_3M.bioMod"

# Load model and pickle
sol1 = get_created_data_from_pickle(pickle_1)
sol2 = get_created_data_from_pickle(pickle_1)
bio_model = BiorbdModel(name_file_model)

# Find index Dofs
dofs_CL = ["Arm", "Forearm", "Thigh", "Leg"]
index_dofs_CL = []
for _, dof in enumerate(dofs_CL):
    index_dofs_CL.append(segment_index(bio_model.model, dof))

time_by_phase_1 = sol1["phase_time"]
time_by_phase_2 = sol2["phase_time"]
for i in range(len(sol1["phase_time"])):
    if i == 0:
        time_by_phase_1[i] = time_by_phase_1[i]
        time_by_phase_2[i] = time_by_phase_2[i]
    else:
        time_by_phase_1[i] = time_by_phase_1[i] + time_by_phase_1[i-1]
        time_by_phase_2[i] = time_by_phase_2[i] + time_by_phase_2[i - 1]


# concatate every "q", "tau", "time"
q_1 = np.concatenate(sol1["q"], axis=1)
q_3 = np.concatenate(sol2["q"], axis=1)
q_2 = np.zeros(shape=(q_3.shape[0], q_3.shape[1]))
tau_1 = np.concatenate(sol1["tau"], axis=1)
tau_3 = np.concatenate(sol2["tau"], axis=1)
tau_2 = np.zeros(shape=(tau_3.shape[0], tau_3.shape[1]))
time_1 = np.concatenate(sol1["time"], axis=0)[:, np.newaxis]
time_2 = np.concatenate(sol2["time"], axis=0)[:, np.newaxis]

# Check if lenght of sol1 and 2 are the same
if q_1.shape[1] == q_2.shape[1]:
    print("Same shape")
else:
    print("Different shape")

# Plot with time
    # tau and q
for dof in range(len(dofs_CL)):

    fig, axs = plt.subplots(2)
    fig.suptitle(dofs_CL[dof])
    axs[0].plot(time_1, q_1[dof].T)
    axs[0].plot(time_2, q_2[dof].T)
    axs[0].set_title("Q", loc="right")
    axs[0].set_xticks([])

    axs[1].plot(time_1, tau_1[dof].T)
    axs[1].plot(time_2, tau_2[dof].T)
    axs[1].set_title("Tau", loc="right")

    # Rajouter trait pour représenter chaque phase
        # Plot chaque phase (time_by phase), barre verticale


plt.show()
plt.savefig("Figures/insoles_position_markers.svg")
plt.clf()

# Plot with normalized time
    # tau and q



# def comparison_simu_CL(path_sol1, path_sol2):
#