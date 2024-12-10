"""
This file is to display the human model into bioviz
"""

from pyorerun import LiveModelAnimation

model_path = "Model2D_7Dof_2C_5M_CL_V3.bioMod"
animation = LiveModelAnimation(model_path, with_q_charts=True)
animation.rerun()
