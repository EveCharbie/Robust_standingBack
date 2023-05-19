import os
import pickle
import numpy as np
from scipy.interpolate import interp1d

import biorbd
import bioviz
import ezc3d


def reorder_markers(markers, model_labels, c3d_marker_labels):

    labels_index = []
    missing_markers_index = []
    for index_model, model_label in enumerate(model_labels):
        missing_markers_bool = True
        for index_c3d, c3d_label in enumerate(c3d_marker_labels):
            if model_label == c3d_label:
                labels_index.append(index_c3d)
                missing_markers_bool = False
        if missing_markers_bool:
            labels_index.append(index_model)
            missing_markers_index.append(index_model)

    markers_reordered = np.zeros((3, 94, markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        if index in missing_markers_index:
            markers_reordered[:, index, :] = np.nan
        else:
            markers_reordered[:, index, :] = markers[:, label_index, :]

    return markers_reordered


def reconstruct_trial(data_filename, model):
    # load the c3d trial
    c3d = ezc3d.c3d(data_filename)
    markers = c3d["data"]["points"][:3, :, :] / 1000  # XYZ1 x markers x time_frame
    c3d_marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"][:94]
    model_labels = [label.to_string() for label in model.markerNames()]
    markers_reordered = reorder_markers(markers, model_labels, c3d_marker_labels)
    # markers_reordered[np.isnan(markers_reordered)] = 0.0  # Remove NaN

    # Dispatch markers in biorbd structure so EKF can use it
    markersOverFrames = []
    for i in range(markers_reordered.shape[2]):
        markersOverFrames.append([biorbd.NodeSegment(m) for m in markers_reordered[:, :, i].T])

    # Create a Kalman filter structure
    frequency = c3d["header"]["points"]["frame_rate"]  # Hz
    # params = biorbd.KalmanParam(frequency=frequency, noiseFactor=1e-10, errorFactor=1e-5)
    params = biorbd.KalmanParam(frequency=frequency)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    def distance_markers(q, *args):
        distances_ignoring_missing_markers = []
        markers_estimated = np.array([marker.to_array() for marker in model.markers(q)]).T
        for i in range(markers_reordered.shape[1]):
            if markers_reordered[0, i, 0] != 0:
                distances_ignoring_missing_markers.append(
                    np.sqrt(np.sum((markers_estimated[:, i] - markers_reordered[:, i, 0]) ** 2))
                )
        return np.sum(distances_ignoring_missing_markers)

    # # Genereate a good guess for the kalman filter
    Q_init = np.zeros(model.nbQ())
    # res = scipy.optimize.minimize(distance_markers, Q_init)
    # Q_init = res.x

    kalman.setInitState(Q_init, np.zeros(model.nbQ()), np.zeros(model.nbQ()))

    # Perform the kalman filter for each frame (the first frame is much longer than the next)
    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)

    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    qdot_recons = np.ndarray((model.nbQdot(), len(markersOverFrames)))
    qddot_recons = np.ndarray((model.nbQddot(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
        qddot_recons[:, i] = Qddot.to_array()

    time_vector = np.arange(0, (len(qddot_recons[0, :]) + 0.5) * 1 / frequency, 1 / frequency)

    return q_recons, qdot_recons, qddot_recons, time_vector


def normalize(param):
    normalize_param = np.zeros(shape=(param.shape[0], param.shape[1]))
    max_param = max(param(axis=0))
    min_param = min(param(axis=0))
    for i in range(param.shape[1]):
        normalize_param[i] = (param[i] - min_param) / (max_param - min_param)
    return normalize_param


def comparison(sol1, sol2, param):

    # Interpolation to have the same shape
    if len(sol1["time_vector"]) > len(sol2["time_vector"]):
        step = sol1["time_vector"][-1] / len(sol1["time_vector"])
        for nb_Dof in range(sol2["param"].shape[0]):
            interpol_time_2 = interp1d(sol2["time_vector"], sol2["param"][nb_Dof, :], kind="linear")
            sol2["time_vector"] = interpol_time_2(np.arange(sol2["param"][0], sol2["param"][-1], step))

    elif len(sol2["time_vector"]) > len(sol1["time_vector"]):
        step = sol2["time_vector"][-1] / len(sol2["time_vector"])
        for nb_Dof in range(sol2["param"].shape[0]):
            interpol_time_2 = interp1d(sol1["time_vector"], sol1["param"][nb_Dof, :], kind="linear")
            sol1["time_vector"] = interpol_time_2(np.arange(sol1["param"][0], sol1["param"][-1], step))
    else:
        pass

    # Time_vector in %
    sol1["norm_time_vector"] = np.arange(0, 100, 100 / len(sol1["time_vector"]))
    sol2["norm_time_vector"] = np.arange(0, 100, 100 / len(sol2["time_vector"]))

    # Normalize data in %

    # Statistics
    # SPM for each Dof and each param(Comparison data 1D)

    return


# --------------------------------------------------------------

FLAG_ANIMATE = False

# load the model
model_path = "EmCo.bioMod"
model = biorbd.Model(model_path)

trials_folder_path = "c3d"
for file in os.listdir(trials_folder_path):

    complete_file = trials_folder_path + "/" + file
    # Kalman filter
    q_recons, qdot_recons, qddot_recons, time_vector = reconstruct_trial(complete_file, model)

    # Inverse dynamics - TODO: Anais
    # Tau vs closed loop -> Tau
    Tau = model.InverseDynamics(q_recons, qdot_recons, qddot_recons)

    # Save the results
    save_path = "reconstructions/" + file[:-4] + ".pkl"
    with open(save_path, "wb") as f:
        data = {
            "q_recons": q_recons,
            "qdot_recons": qdot_recons,
            "qddot_recons": qddot_recons,
            "tau_estimate": Tau,
            "time_vector": time_vector,
        }
        pickle.dump(data, f)

    if FLAG_ANIMATE:
        b = bioviz.Viz(loaded_model=model)
        b.load_movement(q_recons)
        b.exec()
