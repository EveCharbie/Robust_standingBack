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
    InertiaParameters,
    Translations,
    Rotations,
)

""" 
Second method: Create a 2D model from a old 3D model in a txt file
"""


#Create class
class ActuatorConstant:
    def __init__(self, label: str, dof: str, direction: str, Tmax: float):
        self.label = label
        self.dof = dof
        self.direction = direction
        self.Tmax = Tmax * 2

    def __str__(self):
        rt = f"actuator\t {self.label}\n"
        rt += f"\ttype\tConstant\n"
        rt += f"\tdof\t{self.dof}\n"
        rt += f"\tdirection\t{self.direction}\n"
        rt += f"\tTmax\t{self.Tmax}\n"
        rt += f"endactuator\n"

        return rt


class ActuatorGauss3P:
    def __init__(self, label: str, dof: str, direction: str, Tmax: float, T0: float,
                 wmax: float, wc: float, amin: float, wr: float, w1: float, r: float, qopt: float):
        self.label = label
        self.dof = dof
        self.direction = direction
        self.Tmax = Tmax * 2
        self.T0 = T0 * 2
        self.wmax = wmax
        self.wc = wc
        self.amin = amin
        self.wr = wr
        self.w1 = w1
        self.r = r
        self.qopt = qopt

    def __str__(self):
        rt = f"actuator\t {self.label}\n"
        rt += f"\ttype\tGauss3p\n"
        rt += f"\tdof\t{self.dof}\n"
        rt += f"\tdirection\t{self.direction}\n"
        rt += f"\tTmax\t{self.Tmax}\n"
        rt += f"\tT0\t{self.T0}\n"
        rt += f"\twmax\t{self.wmax}\n"
        rt += f"\twc\t{self.wc}\n"
        rt += f"\tamin\t{self.amin}\n"
        rt += f"\twr\t{self.wr}\n"
        rt += f"\tw1\t{self.w1}\n"
        rt += f"\tr\t{self.r}\n"
        rt += f"\tqopt\t{self.qopt}\n"
        rt += f"endactuator\n"

        return rt


# Load the 3D model
model3D = biorbd.Model('/home/lim/Documents/Anais/Robust_standingBack/Pyomecaman_original.bioMod')
# model2D = biorbd.Model('/home/lim/Documents/Anais/Robust_standingBack/Model2D')

# Create a txt file
Model2D = open("Model2D", "w")

# Create list
meshfile_name = ["mesh/pelvis.stl", "mesh/thorax.stl", "mesh/head.stl", "mesh/arm.stl", "mesh/fore_arm.stl",
                 "mesh/hand.stl", "mesh/thigh.stl", "mesh/leg_right.stl", "mesh/foot.stl"]

xyz_segment = [[0, 0, 0.8], [0.0000000000, -0.0515404739, 0.1813885235], [0.0000000000, 0.0435036145, 0.3479414452],
               [-0.091447676, 0.040607449, -0.104557232]]
segment = ["Pelvis", "Thorax", "Head", "Arm", "Forearm", "Hand", "Thigh", "Leg", "Foot"]
parent = ["Pelvis", "Thorax", "Arm", "Forearm", "Pelvis", "Thigh", "Leg"]

List_actuator = [
    ["Thigh", "Leg", "Foot"],
    ["RotX", "RotX", "RotX", "RotX", "RotX", "RotX"],
    ["positive", "negative", "positive", "negative", "positive", "negative"],
    [220.3831, 490.5938, 367.6643, 177.9694, 53.8230, 171.9903],
    [157.4165, 387.3109, 275.0726, 127.1210, 37.2448, 122.8502],
    [475.0000, 562.5000, 1437.5000, 950.0000, 2375.0000, 2000.0000],
    [190.0000, 225.0000, 575.0000, 380.0000, 375.0000, 800.0000],
    [0.9900, 0.9692, 0.9900, 0.9900, 0.9263, 0.9900],
    [40.0000, 40.0000, 40.0000, 40.0000, 40.0000, 40.0000],
    [-90.0000, -90.0000, -90.0000, -89.9997, -90.0000, -90.0000],
    [56.4021, 48.6999, 31.7218, 57.0370, 58.9832, 21.8717],
    [25.6939, 72.5836, 61.7303, 33.2908, 0.7442, 12.6824]
]

# Write on the txt file
Model2D.write("version 4\n\ngravity 0 0 -9.81\n\n")

# Pelvis segment
Model2D.write("segment\t" + segment[0] + "\n")
Model2D.write("\tRT -0.1 0 0\txyz" + " " + str(xyz_segment[0][0]) + " " + str(xyz_segment[0][1]) + " " + str(
    xyz_segment[0][0]) + "\n")
Model2D.write("\ttranslations yz\n")
Model2D.write("\trotation\t x\n")
Model2D.write("\trangesQ\n\t\t" + str(model3D.segments()[0].QRanges()[0].min()) + " " + str(
    model3D.segments()[0].QRanges()[0].max()) + "\n\t\t "
              + str(model3D.segments()[0].QRanges()[1].min()) + " " + str(
    model3D.segments()[0].QRanges()[1].max()) + "\n\t\t"
              + str(model3D.segments()[0].QRanges()[2].min()) + " " + str(
    model3D.segments()[0].QRanges()[2].max()) + "\n")
Model2D.write("\trangesQdot\t" + str(model3D.segments()[0].QDotRanges()[0].min()) + " " + str(
    model3D.segments()[0].QDotRanges()[0].max()) + "\n\t\t"
              + str(model3D.segments()[0].QDotRanges()[1].min()) + " " + str(
    model3D.segments()[0].QDotRanges()[1].max()) + "\n\t\t"
              + str(model3D.segments()[0].QDotRanges()[2].min()) + " " + str(
    model3D.segments()[0].QDotRanges()[2].max()) + "\n")
Model2D.write("\tmass\t" + str(model3D.segments()[0].characteristics().mass()) + "\n")
Model2D.write("\tinertia\n\t\t" + str(model3D.segments()[0].characteristics().inertia().to_array()[0][0]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[0][1]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[0][2]) + "\n\t\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[1][0]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[1][1]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[1][2]) + "\n\t\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[2][0]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[2][1]) + "\t"
              + str(model3D.segments()[0].characteristics().inertia().to_array()[2][2]) + "\n")
Model2D.write("\tcom\t"
              + str(model3D.segments()[0].characteristics().CoM().to_array()[0]) + "\t"
              + str(model3D.segments()[0].characteristics().CoM().to_array()[1]) + "\t"
              + str(model3D.segments()[0].characteristics().CoM().to_array()[2]) + "\n")
Model2D.write("\tmeshfile\t" + meshfile_name[0] + "\n")
Model2D.write("endsegment\n\n")

#Actuator Pelvis
Model2D.write(str(ActuatorConstant(segment[0], "TransY", "positive", 0.0)))
Model2D.write(str(ActuatorConstant(segment[0], "TransY", "negative", 0.0)))
Model2D.write(str(ActuatorConstant(segment[0], "TransZ", "positive", 0.0)))
Model2D.write(str(ActuatorConstant(segment[0], "TransZ", "negative", 0.0)))

# Thorax and head segment
for i in range(1, 3):
    Model2D.write("segment\t" + segment[i] + "\n")
    Model2D.write("\tparent\t" + parent[i - 1] + "\n")
    Model2D.write("\tRT 0 0 0\txyz" + " " + str(xyz_segment[i][0]) + " " + str(xyz_segment[i][1]) + " " + str(
        xyz_segment[i][2]) + "\n")
    Model2D.write("\tmass\t" + str(model3D.segments()[i].characteristics().mass()) + "\n")
    Model2D.write("\tinertia\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][0]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][1]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][2]) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][0]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][1]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][2]) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][0]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][1]) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][2]) + "\n")
    Model2D.write("\tcom\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[0]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[1]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[2]) + "\n")
    Model2D.write("\tmeshfile\t" + meshfile_name[i] + "\n")
    Model2D.write("endsegment\n\n")

# Arm

Model2D.write("segment\t" + segment[3] + "\n")
Model2D.write("\tparent\t" + parent[3 - 2] + "\n")
Model2D.write("\tRTinMatrix\t1\n")
Model2D.write("\tRT\n\t\t" + "1" + " " + " 0 " + " " + "0" + " "
              + str((model3D.localJCS()[3].to_array()[0][3] + model3D.localJCS()[6].to_array()[0][3]) / 2) + "\n\t\t"
              + "0" + " " + "1" + " " + "0" + " " + str((model3D.localJCS()[3].to_array()[1][3] + model3D.localJCS()[6].to_array()[1][3]) / 2) + "\n\t\t"
              + "0" + " " + "0 " + " " + "1" + " "
              + str((model3D.localJCS()[3].to_array()[2][3] + model3D.localJCS()[6].to_array()[2][3]) / 2) + "\n\t\t"
              + "0" + " " + "0" + " " + "0" + " " + "1" + "\n")
Model2D.write("\trangesQ\t" + str(model3D.segments()[3].QRanges()[0].min()) + " " + str(
    model3D.segments()[3].QRanges()[0].max()) + "\n")
Model2D.write("\trangesQdot\t" + str(model3D.segments()[3].QDotRanges()[0].min()) + " " + str(
    model3D.segments()[3].QDotRanges()[0].max()) + "\n")
Model2D.write("\tmass\t" + str(model3D.segments()[3].characteristics().mass() * 2) + "\n")
Model2D.write("\tinertia\n\t\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[0][0] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[0][1] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[0][2] * 2) + "\n\t\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[1][0] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[1][1] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[1][2] * 2) + "\n\t\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[2][0] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[2][1] * 2) + "\t"
              + str(model3D.segments()[3].characteristics().inertia().to_array()[2][2] * 2) + "\n")
Model2D.write("\tcom\t"
              + str(model3D.segments()[3].characteristics().CoM().to_array()[0]) + "\t"
              + str(model3D.segments()[3].characteristics().CoM().to_array()[1]) + "\t"
              + str(model3D.segments()[3].characteristics().CoM().to_array()[2]) + "\n")
Model2D.write("\tmeshfile\t" + meshfile_name[3] + "\n")
Model2D.write("endsegment\n\n")

#Model2D.write(str(ActuatorGauss3P(segment[3], "RotX", "positive", 112.8107, 89.0611, 1000, 400, 0.878, 40, -6.275, 109.6679, -41.0307)))
#Model2D.write(str(ActuatorGauss3P(segment[3], "RotX", "negative", 162.7655, 128.4991, 812.5000, 325.0000, 0.9678, 40.0000, -90.0000, 103.9095, -101.6627)))

# Forearm and hand
for i in range(4, 6):
    Model2D.write("segment\t" + segment[i] + "\n")
    Model2D.write("\tparent\t" + parent[i - 2] + "\n")
    Model2D.write("\tRTinMatrix\t1\n")
    Model2D.write("\tRT\n\t\t" + "1" + " " + " 0 " + " " + "0" + " "
                  + str((model3D.localJCS()[i].to_array()[0][3] + model3D.localJCS()[i + 3].to_array()[0][3]) / 2) + "\n\t\t"
                  + "0" + " " + "1" + " " + "0" + " " + str((model3D.localJCS()[i].to_array()[1][3] + model3D.localJCS()[i + 3].to_array()[1][3]) / 2) + "\n\t\t"
                  + "0" + " " + "0 " + " " + "1" + " "
                  + str((model3D.localJCS()[i].to_array()[2][3] + model3D.localJCS()[i + 3].to_array()[2][3]) / 2) + "\n\t\t"
                  + "0" + " " + "0" + " " + "0" + " " + "1" + "\n")
    Model2D.write("\tmass\t" + str(model3D.segments()[i].characteristics().mass() * 2) + "\n")
    Model2D.write("\tinertia\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][2] * 2) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][2] * 2) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][2] * 2) + "\n")
    Model2D.write("\tcom\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[0]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[1]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[2]) + "\n")
    Model2D.write("\tmeshfile\t" + meshfile_name[i] + "\n")
    Model2D.write("endsegment\n\n")

# Thigh, Leg, foot
for i in range(10, 13):
    Model2D.write("segment\t" + segment[i - 4] + "\n")
    Model2D.write("\tparent\t" + parent[i - 6] + "\n")
    Model2D.write("\tRTinMatrix\t1\n")
    Model2D.write("\tRT\n\t\t" + "1" + " " + " 0 " + " " + "0" + " "
                  + str((model3D.localJCS()[i].to_array()[0][3] + model3D.localJCS()[i + 4].to_array()[0][3]) / 2) + "\n\t\t"
                  + "0" + " " + "1" + " " + "0" + " " + str((model3D.localJCS()[i].to_array()[1][3] + model3D.localJCS()[i + 4].to_array()[1][3]) / 2) + "\n\t\t"
                  + "0" + " " + "0 " + " " + "1" + " "
                  + str((model3D.localJCS()[i].to_array()[2][3] + model3D.localJCS()[i + 4].to_array()[2][3]) / 2) + "\n\t\t"
                  + "0" + " " + "0" + " " + "0" + " " + "1" + "\n")
    Model2D.write("\trangesQ\t" + str(model3D.segments()[i].QRanges()[0].min()) + " " + str(
        model3D.segments()[i].QRanges()[0].max()) + "\n")
    Model2D.write("\trangesQdot\t" + str(model3D.segments()[i].QDotRanges()[0].min()) + " " + str(
        model3D.segments()[i].QDotRanges()[0].max()) + "\n")
    Model2D.write("\tmass\t" + str(model3D.segments()[i].characteristics().mass() * 2) + "\n")
    Model2D.write("\tinertia\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[0][2] * 2) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[1][2] * 2) + "\n\t\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][0] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][1] * 2) + "\t"
                  + str(model3D.segments()[i].characteristics().inertia().to_array()[2][2] * 2) + "\n")
    Model2D.write("\tcom\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[0]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[1]) + "\t"
                  + str(model3D.segments()[i].characteristics().CoM().to_array()[2]) + "\n")
    Model2D.write("\tmeshfile\t" + meshfile_name[i - 4] + "\n")
    Model2D.write("endsegment\n\n")

    """
    Model2D.write(str(ActuatorGauss3P(
        str(List_actuator[0][i-10]),
        str(List_actuator[1][i-10]),
        str(List_actuator[2][i-10]),
        float(List_actuator[3][i-10]),
        float(List_actuator[4][i-10]),
        float(List_actuator[5][i-10]),
        float(List_actuator[6][i-10]),
        float(List_actuator[7][i-10]),
        float(List_actuator[8][i-10]),
        float(List_actuator[9][i-10]),
        float(List_actuator[10][i-10]),
        float(List_actuator[11][i-10])
    )))
    Model2D.write(str(ActuatorGauss3P(
        str(List_actuator[0][i-10]),
        str(List_actuator[1][i-9]),
        str(List_actuator[2][i-9]),
        float(List_actuator[3][i-9]),
        float(List_actuator[4][i-9]),
        float(List_actuator[5][i-9]),
        float(List_actuator[6][i-9]),
        float(List_actuator[7][i-9]),
        float(List_actuator[8][i-9]),
        float(List_actuator[9][i-9]),
        float(List_actuator[10][i-9]),
        float(List_actuator[11][i-9])
    )))
    """

# Close the txt file
Model2D.close()

# Open the new model
#model2D = biorbd.Model('/home/lim/Documents/Anais/Robust_standingBack/Model2D')

# Visualisation of the new model
# b = bioviz.Viz('/home/lim/Documents/Anais/Robust_standingBack/Model2D')
# b.exec()
# b = bioviz.Viz('/home/lim/Documents/Anais/Robust_standingBack/Pyomecaman_original.bioMod')
