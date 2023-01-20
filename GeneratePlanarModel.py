import biorbd
import numpy as np
import bioviz
from biorbd.model_creation import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    C3dData,
    Marker,
    MarkerReal,
    Mesh,
    MeshReal,
    Segment,
    SegmentReal,
    SegmentCoordinateSystemReal,
    SegmentCoordinateSystem,
    InertiaParametersReal,
    Translations,
    Rotations,
)

# Visualization of the model 3D
# Model3D = bioviz.Viz('/home/lim/Documents/Anais/Robust_standingBack/Pyomecaman_original.bioMod')
model3D = biorbd.Model('/home/lim/Documents/Anais/Robust_standingBack/Pyomecaman_original.bioMod')

#path_model3D= '/home/lim/Documents/Anais/Robust_standingBack/Pyomecaman_original.bioMod'
#def model_2d_from_model_3d(path_model3D):
#model3D = biorbd.Model(path_model3D)

"""
We create a new model in 2D (sagittal plane) from an old model
"""

# Fill the kinematic chain model

model2D = BiomechanicalModelReal()

# The trunk segment
model2D["Pelvis"] = SegmentReal(
    name="Pelvis",
    translations=Translations.XY,
    rotations=Rotations.Z,
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[0].characteristics().mass(),
        center_of_mass=model3D.segments()[0].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[0].characteristics().inertia().to_array()))

for i in range(0, 5):
    model2D["Pelvis"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                            parent_name="Pelvis",
                                            position=model3D.markers()[i].to_array()))

# The Thorax segment
model2D["Thorax"] = SegmentReal(
    parent_name="Pelvis",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (.0000000000, -0.0515404739, 0.1813885235)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[1].characteristics().mass(),
        center_of_mass=model3D.segments()[1].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[1].characteristics().inertia().to_array()))

for i in range(6, 11):
    model2D["Thorax"].add_marker(MarkerReal(name=model3D.markerNames()[6].to_string(),
                                            parent_name="Thorax",
                                            position=model3D.markers()[6].to_array()))

# The Head segment
model2D["Head"] = SegmentReal(
    parent_name="Thorax",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (.0000000000, -0.0515404739, 0.1813885235)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[2].characteristics().mass(),
        center_of_mass=model3D.segments()[2].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[2].characteristics().inertia().to_array()))

for i in range(12, 16):
    model2D["Head"].add_marker(MarkerReal(name=model3D.markerNames()[12].to_string(),
                                          parent_name="Head",
                                          position=model3D.markers()[12].to_array()))

# The arm segment
model2D["Arm"] = SegmentReal(
    parent_name="Thorax",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (0, 0, 0.53)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[3].characteristics().mass()*2,
        center_of_mass=model3D.segments()[3].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[3].characteristics().inertia().to_array()*2))

for i in range(17, 21):
    model2D["Arm"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                         parent_name="Arm",
                                         position=(model3D.markers()[i].to_array()[0],
                                                   model3D.markers()[i].to_array()[1],
                                                   (model3D.markers()[i].to_array()[2] + model3D.markers()[i + 16].to_array()[2]) / 2)))
# The forearm segment
model2D["Forearm"] = SegmentReal(
    name="Forearm",
    parent_name="Arm",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (0, 0, -0.28)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[4].characteristics().mass()*2,
        center_of_mass=model3D.segments()[4].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[4].characteristics().inertia().to_array()*2))

for i in range(22, 29):
    model2D["Forearm"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                             parent_name="Forearm",
                                             position=(model3D.markers()[i].to_array()[0],
                                                       model3D.markers()[i].to_array()[1],
                                                       (model3D.markers()[i].to_array()[2] + model3D.markers()[i+16].to_array()[2]) / 2)))

# The hand segment
model2D["Hand"] = SegmentReal(
    name="Hand",
    parent_name="Forearm",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (0, 0, -0.27)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[5].characteristics().mass() * 2,
        center_of_mass=model3D.segments()[5].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[5].characteristics().inertia().to_array() * 2)
)

for i in range(30, 32):
    model2D["Hand"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                          parent_name="Hand",
                                          position=(model3D.markers()[i].to_array()[0],
                                                    model3D.markers()[i].to_array()[1],
                                                    (model3D.markers()[i].to_array()[2] + model3D.markers()[i + 16].to_array()[2]) / 2)))

# The thigh segment
model2D["Thigh"] = SegmentReal(
    name="Thigh",
    parent_name="Pelvis",
    rotations=Rotations.Z,
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[9].characteristics().mass()*2,
        center_of_mass=model3D.segments()[9].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[9].characteristics().inertia().to_array()*2))

for i in range(49, 53):
    model2D["Thigh"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                           parent_name="Thigh",
                                           position=(model3D.markers()[i].to_array()[0],
                                                     model3D.markers()[i].to_array()[1],
                                                     (model3D.markers()[i].to_array()[2] + model3D.markers()[i + 17].to_array()[2]) / 2)))
# The leg segment
model2D["Leg"] = SegmentReal(
    name="Leg",
    parent_name="THIGH",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (0, 0, 0), "xyz", (0, 0, -0.42)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[10].characteristics().mass()*2,
        center_of_mass=model3D.segments()[10].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[10].characteristics().inertia().to_array()*2))

for i in range(54, 59):
    model2D["Leg"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                         parent_name="Leg",
                                         position=(model3D.markers()[i].to_array()[0],
                                                   model3D.markers()[i].to_array()[1],
                                                   (model3D.markers()[i].to_array()[2] + model3D.markers()[i + 17].to_array()[2]) / 2)))

# The foot segment
model2D["Foot"] = SegmentReal(
    name="Foot",
    parent_name="Leg",
    rotations=Rotations.Z,
    segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
        (-np.pi / 2, 0, 0), "xyz", (0, 0, -0.43)),
    inertia_parameters=InertiaParametersReal(
        mass=model3D.segments()[11].characteristics().mass()*2,
        center_of_mass=model3D.segments()[11].characteristics().CoM().to_array()[:, np.newaxis],
        inertia=model3D.segments()[11].characteristics().inertia().to_array()*2))

for i in range(60, 65):
    model2D["Foot"].add_marker(MarkerReal(name=model3D.markerNames()[i].to_string(),
                                          parent_name="Foot",
                                          position=(model3D.markers()[i].to_array()[0],
                                                    model3D.markers()[i].to_array()[1],
                                                    (model3D.markers()[i].to_array()[2] +
                                                    model3D.markers()[i + 17].to_array()[2]) / 2)))

# Put the model together, print it and print it to a bioMod file
model2D.write('/home/lim/Documents/Anais/Robust_standingBack/model2D_V2.bioMod')

#model = biorbd.Model('/home/lim/Documents/Anais/Robust_standingBack/model2D.bioMod')

#assert model2D.nbQ() == 7
#assert model2D.nbSegment() == 9
#assert model2D.nbMarkers() == 41
#np.testing.assert_almost_equal(model2D.markers(np.zeros((model2D.nbQ(),)))[-3].to_array(), [0, 0.25, -0.85], decimal=4)

#if remove_temporary:
#    os.remove(kinematic_model_file_path)




#name = m.segments()[0].name().to_string()
#m.segments()[0].characteristics().CoM().to_array()