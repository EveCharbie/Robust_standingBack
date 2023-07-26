import numpy as np
import pickle
from Save import get_created_data_from_pickle
import matplotlib.pyplot as plt
from bioptim import BiorbdModel
from biorbd import segment_index


def holonomics_constraints_graph(sol, index_holonomics_constraints, lambdas):
    plt.plot(sol.time[index_holonomics_constraints], lambdas[0, :],
             label="y",
             marker="o",
             markersize=5,
             markerfacecolor="blue")
    plt.plot(sol.time[index_holonomics_constraints], lambdas[1, :],
             label="z",
             marker="o",
             markersize=5,
             markerfacecolor="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Lagrange multipliers (N)")
    plt.title("Lagrange multipliers of the holonomic constraint")
    plt.legend()
    plt.show()


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

# Info
pickle_1 = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V13.pkl"
pickle_2 = "/home/mickael/Documents/Anais/Robust_standingBack/holonomic_research/Salto_6phases_V10.pkl"
name_file_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_7Dof_0C_3M.bioMod"

# Load model and pickle
sol1 = get_created_data_from_pickle(pickle_1)
sol2 = get_created_data_from_pickle(pickle_1)
bio_model = BiorbdModel(name_file_model)

# Find index Dofs
# dofs_CL = ["Arm", "Forearm", "Thigh", "Leg"]
dofs_CL = []
for i in range(len(bio_model.segments)):
    dofs_CL.append(bio_model.segments[i].name().to_string())
dofs_CL = dofs_CL[:-1]  # TODO: A changer
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
tau_nan = np.zeros(shape=(3, (q_1.shape[1])))
tau_nan[:] = np.nan
tau_1 = np.concatenate(sol1["tau"], axis=1)
tau_1 = np.concatenate((tau_nan, tau_1), axis=0)
tau_3 = np.concatenate(sol2["tau"], axis=1)
tau_2 = np.zeros(shape=(tau_3.shape[0], tau_3.shape[1]))
tau_2 = np.concatenate((tau_nan, tau_2), axis=0)
time_1 = np.concatenate(sol1["time"], axis=0)[:, np.newaxis]
time_2 = np.concatenate(sol2["time"], axis=0)[:, np.newaxis]

# Check if lenght of sol1 and 2 are the same
if q_1.shape[1] == q_2.shape[1]:
    print("Same shape")
else:
    print("Different shape")

#Recap comparison



# Plot with time
    # tau and q

fig, axs = plt.subplots(nrows=len(dofs_CL), ncols=2, sharex="row")
# fig.suptitle(dofs_CL[dof])
for dof in range(len(dofs_CL)):
    axs[dof, 0].plot(time_1, q_1[dof].T, color='b')
    axs[dof, 0].plot(time_2, q_2[dof].T, color='r')
    axs[dof, 0].text(-0.15, 0.5, str(dofs_CL[dof]),
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical',
            transform=axs[dof, 0].transAxes)

    # axs[0].set_title("Q", loc="right")
    if dof < len(dofs_CL) - 1:
        axs[dof, 0].set_xticks([])

    axs[dof, 1].step(time_1, tau_1[dof].T, color='b')
    axs[dof, 1].step(time_2, tau_2[dof].T, color='r')

    axs[0, 1].set_title('Tau')
    axs[0, 0].set_title('Q')

    # Add vertical line for phases
    for phase1 in range(len(time_by_phase_1)):
        axs[dof, 0].axvline(x=time_by_phase_1[phase1], ymax=0.05, color='b', label='Phase ' + str(phase1))
        if dof > 3:
            axs[dof, 1].axvline(x=time_by_phase_1[phase1], ymax=0.05, color='b', label='Phase ' + str(phase1))
    for phase2 in range(len(time_by_phase_2)):
        axs[dof, 0].axvline(x=time_by_phase_2[phase2], ymax=0.05, color='r', label='Phase ' + str(phase2))
        if dof > 3:
            axs[dof, 1].axvline(x=time_by_phase_2[phase2], ymax=0.05, color='r', label='Phase ' + str(phase2))

    # axs[dof, 1].set_title("Tau", loc="right")
    if dof < len(dofs_CL) - 1:
        axs[dof, 1].set_xticks([])
plt.show()
# plt.savefig("Figures/Comparison_tau_q_" + str(dofs_CL[dof])+".svg")
# plt.clf()

# Plot with normalized time
# time1_normalized = []
#
# for i in range(len(time_1)):
#     time1_normalized.append(time_1[i]*100/time_1[-1])
#
# time2_normalized = []
# for i in range(len(time_2)):
#     time2_normalized.append(time_2[i] * 100 / time_2[-1])
#
# time_by_phase_1_normalized = []
# time_by_phase_2_normalized = []
# for nb_phase in range(len(time_by_phase_1)):
#     time_by_phase_1_normalized.append(time_by_phase_1[nb_phase] * 100 / time_1[-1])
#     time_by_phase_2_normalized.append(time_by_phase_2[nb_phase] * 100 / time_1[-1])
#
#     # tau and q
# for dof in range(len(dofs_CL)):
#     fig, axs = plt.subplots(2)
#     fig.suptitle(dofs_CL[dof])
#     axs[0].plot(time1_normalized, q_1[dof].T, color='b')
#     axs[0].plot(time2_normalized, q_2[dof].T, color='r')
#     axs[0].set_title("Q", loc="right")
#     axs[0].set_xticks([])
#
#     axs[1].plot(time1_normalized, tau_1[dof].T, color='b')
#     axs[1].plot(time2_normalized, tau_2[dof].T, color='r')
#     # Add vertical line for phases
#     for phase1 in range(len(time_by_phase_1)):
#         axs[0].axvline(x=time_by_phase_1_normalized[phase1], color='b', label='Phase ' + str(phase1))
#         axs[1].axvline(x=time_by_phase_1_normalized[phase1], color='b', label='Phase ' + str(phase1))
#     for phase2 in range(len(time_by_phase_2)):
#         axs[0].axvline(x=time_by_phase_2_normalized[phase2], color='r', label='Phase ' + str(phase2))
#         axs[1].axvline(x=time_by_phase_2_normalized[phase2], color='r', label='Phase ' + str(phase2))
#     axs[1].set_title("Tau", loc="right")
#     plt.show()
#     plt.savefig("Figures/Comparison_normalized_tau_q_" + str(dofs_CL[dof])+".svg")
#     plt.clf()
#

# def comparison_simu_CL(path_sol1, path_sol2):


# Ajouter toutes ces infos dans un .txt + Ajouter SD
key = ["q", "qdot", "tau"]
bio_model = BiorbdModel(name_file_model)
dofs_CL = ["Arm", "Forearm", "Thigh", "Leg"]
index_dofs_CL = []
for _, dof in enumerate(dofs_CL):
    index_dofs_CL.append(segment_index(bio_model.model, dof))

print("sol1: " + str(pickle_1))
print("sol1: " + str(pickle_2) + "\n")

for i in range(len(key)):
    for nb_phase in range(len(sol1["tau"])):
        print("\nPhase " + str(nb_phase) + "\t" + str(key[i]) + "\n")
        print("\t" + "sol1\t" + "sol2\n")
        for dof in range(len(dofs_CL)):
            if key[i] == "tau":
                valeursol1 = np.nanmean(sol1[key[i]][nb_phase][index_dofs_CL[dof]-3])
                valeursol2 = np.nanmean(sol2[key[i]][nb_phase][index_dofs_CL[dof]-3])
            else:
                valeursol1 = np.nanmean(sol1[key[i]][nb_phase][index_dofs_CL[dof]])
                valeursol2 = np.nanmean(sol2[key[i]][nb_phase][index_dofs_CL[dof]])

            print(str(dofs_CL[dof]) + "\t" + str(valeursol1) + "\t" + str(valeursol2))