import bioviz
import pickle
import sys
sys.path.append("/home/lim/Documents/Anais/bioviz")
sys.path.append("/home/lim/Documents/Anais/bioptim")

#name_file_pickle = "/home/lim/Anais/Results_opti/Data/Salto_6phases/Salto_6phases_V14"
#name_file = "/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod"
# def get_pickle(file: str):
#       with open(file, "rb") as file:
#             while True:
#                   try:
#                         data_tmp = pickle.load(file)
#                   except:
#                         break
#

def get_created_data_from_pickle(file: str):
      with open(file, "rb") as file:
            while True:
                  try:
                        data_tmp = pickle.load(file)
                  except:
                        break

      datas_q = data_tmp["states"][1]
      #datas_qdot = data_tmp["states"]["qdot"]
      #datas_tau = data_tmp["controls"]["tau"]
      #data_status = data_tmp["status"]
      #data_mus = data_tmp["controls"]["muscles"]
      #data_time = data_tmp["real_time_to_optimize"]
      #data_it = data_tmp["iterations"]
      #data_cost = data_tmp["detailed_cost"]

      return datas_q #datas_qdot, datas_tau, data_status, data_mus, data_it, data_time, data_cost

q = get_created_data_from_pickle("/home/lim/Anais/Results_opti/Data/Salto_6phases/Salto_6phases_V15.pkl")
#b = bioviz.Viz("/home/lim/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M.bioMod", show_floor=True, show_meshes=False)
#b.load_movement(q)
#b.exec()