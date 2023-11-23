import pickle
import numpy as np

# --- Functions --- #
def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas_q = data_tmp["q"]
    datas_qdot = data_tmp["qdot"]
    datas_tau = data_tmp["tau"]
    # data_status = data_tmp["status"]
    # data_mus = data_tmp["controls"]["muscles"]
    data_time = data_tmp["real_time_to_optimize"]
    # data_it = data_tmp["iterations"]
    # data_cost = data_tmp["detailed_cost"]
    data_time_node = data_tmp["time"]

    return datas_q, datas_qdot, datas_tau, data_time, data_time_node

# --- Script --- #
name_file = "/home/mickael/Documents/Anais/Robust_standingBack/Code - examples/Jump-salto/Salto_close_loop_with_pelvis_3phases_V1.pkl"
q, q_dot, tau, time, time_node = get_created_data_from_pickle(name_file)


