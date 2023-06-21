import numpy as np


def constraints_graphs(ocp, sol):
    """Plot the graphs of the constraints"""

    simple_derivative0 = []
    simple_derivative1 = []
    for qi, qdoti in zip(sol.states["q"].T, sol.states["qdot"].T):
        print(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti))
        simple_derivative0.append(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti)[0].toarray()[0, 0])
        simple_derivative = np.array(simple_derivative0).squeeze()
        simple_derivative1.append(ocp.nlp[0].model.holonomic_constraints_derivative(qi, qdoti)[1].toarray()[0, 0])
        simple_derivative = np.array(simple_derivative1).squeeze()

    double_derivative_0 = []
    double_derivative_1 = []
    # print(sol.controls["tau"])
    for qi, qdoti, taui in zip(sol.states["q"].T, sol.states["qdot"].T, sol.controls["tau"].T):
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
    ax[0].plot(sol.time, ocp.nlp[0].model._holonomic_constraints[0](sol.states["q"]).T)
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
