import bioviz
import pickle
import sys
sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/lim/Documents/Anais/bioptim")


def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """

    data = {}
    q = []
    qdot = []
    tau = []
    # data = dict(states=sol.states,
    #             controls=sol.controls,
    #             parameters=sol.parameters,
    #             iterations=sol.iterations,
    #             cost=sol.cost,
    #             detailed_cost=sol.detailed_cost,
    #             real_time_to_optimize=sol.real_time_to_optimize,
    #             status=sol.status)

    for i in range(len(sol.states)):
        q.append(sol.states[i]["q"])
        qdot.append(sol.states[i]["qdot"])
        tau.append(sol.controls[i]["tau"])

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)

# def save graph

#def save_video()
