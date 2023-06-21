import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
import casadi as cas
import numpy as np
from casadi import MX, vertcat, Function

#--- Parameters ---#
l1 = 1
l2 = 1
xp = 0
yp = 1

#--- Test ---#

# Fonction angle theta1 et theta2
theta2 = cas.acos(
    (xp ** 2 + yp ** 2 - (l2 ** 2 + l1 ** 2)) / 2 * l1 * l2
)

# theta1 = cas.atan(markers_frame_thorax[2]/markers_frame_thorax[1]) - cas.atan(
#     (L2 * cas.sin(theta2)) / (L1 + L2 * cas.cos(theta2))
# ) # TODO: tan2
# theta1 = cas.atan2((yp / xp),
#                    ((l2 * cas.sin(theta2)) / (l1 + l2 * cas.cos(theta2)))
#                    )

theta1 = cas.atan2(
    (-xp * l2 * cas.sin(theta2) + yp * (l1 + l2 * cas.cos(theta2))),
    (xp * (l1 + l2 * cas.cos(theta2)) + yp * l2 * cas.sin(theta2))
)

# theta1 = cas.atan2(
#     (yp - cas.sqrt(yp ** 2 + xp ** 2 - ((xp ** 2 + yp ** 2 + l1 ** 2 - l2 ** 2)/(2 * l1)))),
#     (xp + ((xp ** 2 + yp ** 2 + l1 ** 2 - l2 ** 2)/(2 * l1)))
# )

print("Theta 1:" + str(theta1*180/np.pi))
print("Theta 2:" + str(theta2*180/np.pi))


# # pour tester que l'equation du markers ne depend des q non associés à cette branche.
name_file_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_4Dof_0C_5M_CL_modif.bioMod"
model = biorbd_eigen.Model(name_file_model)
q_num = np.array([1, 2, 3, 0, 0, 5, 6])
# #Recuperer le marqueur de la jambe    (marker [2])
pos_marker_knee = model.markers(q_num)[2].to_array()
print("Position des marqueurs du genou avec q égal à " + str(q_num) + " :" + str(pos_marker_knee))
# # verifier que la position du marqueur dans le repere global ne change pas quand tu change la valeurs des zeros
q2 = np.array([1, 2, 3, 2, 2, 5, 6])
pos_marker_knee2 = model.markers(q_num)[2].to_array()
print("Position des marqueurs du genou avec q égal à " + str(q2) + " :" + str(pos_marker_knee2))


# # Idem mais en casadi
model_mx = biorbd.Model(name_file_model)
u = MX.sym("u", 5, 1)
# index_marqueur_jambe = 1

# # second temps envoyer des variables v à la place de zeros
v = MX.sym("v", 2, 1)
# q = vertcat(u[:3], MX(0), MX(0), u[3:])
q = vertcat(u[:3], v, u[3:])

# my_marker = m_mx.markers(q)[index_marqueur_jambe]
pos_marker_knee_mx = model_mx.markers(q)[2].to_mx()
print(pos_marker_knee_mx)
# # Je dois m'assurer que je peux créer la fonction casadi
f_func = Function("marker", [u], [pos_marker_knee_mx])
# # il faut que tu t'assure que ça renvoie la meme fvaleur dans la version eigen (pas casadi)
f_func(np.concatenate([q_num[:3], q_num[5:]]))
print(f_func)
print(f_func(np.concatenate([q_num[:3], q_num[5:]])))
