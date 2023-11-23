import casadi as cas
import numpy as np
from ocp_example_2 import generate_close_loop_constraint
from biorbd_model_holonomic import BiorbdModelCustomHolonomic
import bioviz

# --- Parameters --- #
rotation_pelvis = True

# ------------------ #

if rotation_pelvis is True:
    # BioModel path
    model_path = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_4Dof_0C_5M_CL_V2.bioMod"
    bio_model = BiorbdModelCustomHolonomic(model_path)

    # Made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model,
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )

    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model.set_dependencies(independent_joint_index=[2, 3], dependent_joint_index=[0, 1])
    pose_salto_tendu = [2.19, -1.19]
    pose_salto_groupe = [2.6, -2.17]

    u_1 = np.arange(start=pose_salto_tendu[0], stop=pose_salto_groupe[0], step=abs(pose_salto_groupe[0]-pose_salto_tendu[0])/20)
    u_2 = np.arange(start=pose_salto_tendu[1], stop=pose_salto_groupe[1], step=-abs(pose_salto_groupe[1]-pose_salto_tendu[1])/20)
    u = np.concatenate((u_1[:, np.newaxis], u_2[:, np.newaxis]), axis=1)

    q = np.zeros((bio_model.nb_tau, u.shape[0]))
    for i, ui in enumerate(u):
        vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

    # ui = np.array(pose_salto_groupe)[:, np.newaxis]
    # vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
    # qi = bio_model.q_from_u_and_v(ui, vi).toarray().squeeze()
    # viz = bioviz.Viz(model_path)
    # viz.load_movement(qi[:, np.newaxis])
    # viz.exec()

else:
    # BioModel path
    model_path = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_4Dof_0C_5M_CL_V2.bioMod"
    bio_model = BiorbdModelCustomHolonomic(model_path)

    # Made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model,
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )

    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model.set_dependencies(independent_joint_index=[2, 3], dependent_joint_index=[0, 1])
    pose_salto_tendu = [2.19, -1.19]
    pose_salto_groupe = [2.6, -2.17]

    u_1 = np.arange(start=pose_salto_tendu[0], stop=pose_salto_groupe[0],
                    step=abs(pose_salto_groupe[0] - pose_salto_tendu[0]) / 20)
    u_2 = np.arange(start=pose_salto_tendu[1], stop=pose_salto_groupe[1],
                    step=-abs(pose_salto_groupe[1] - pose_salto_tendu[1]) / 20)
    u = np.concatenate((u_1[:, np.newaxis], u_2[:, np.newaxis]), axis=1)

    q = np.zeros((bio_model.nb_tau, u.shape[0]))
    for i, ui in enumerate(u):
        vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

    # ui = np.array(pose_salto_groupe)[:, np.newaxis]
    # vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
    # qi = bio_model.q_from_u_and_v(ui, vi).toarray().squeeze()
    # viz = bioviz.Viz(model_path)
    # viz.load_movement(qi[:, np.newaxis])
    # viz.exec()


