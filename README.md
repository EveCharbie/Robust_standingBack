# Robust_standingBack
This repo aims to generate robust techniques for a standing back somersault on ground.

## Steps
In first, I create codes to generate simple jump in two (`Salto_2phases.py`) and 
three phases(`Salto_3phases.py`) in order to learn how to use bioptim.

The second step was to add a backward somersault during the flight phase, when the model attain the maximal height of 
his centre of mass (CoM). I begin by implement a backward somersault in a 3 phases movement (`Salto_3phases.py`) 
and gradually I increase the number of phases until reach 6 phases(`Salto_6phases.py`).

Add step actuator


## Model

Create model 2D from 3D model with `GeneratePlanarModel.py`.


The different model use are in the folder "Model". 
The name of the model is defines like this: `"Model2D_8Dof_2C_3M"` according to the type of the model (i.e. 2D or 3D), the number of degree of freedom (i.e. Dof) 
the number of contact (i.e. C), the number of markers (i.e. M)

## How to save data

To save data, we use `Save.py`. 


##

