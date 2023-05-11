# Robust_standingBack
This repo aims to generate robust techniques for a standing back somersault on ground.

## Steps
### 2D model
Create model 2D from 3D model with `GeneratePlanarModel.py`.


### Generate jump
In first, I create codes to generate simple jump in two ([`Salto_2phases.py`](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/Jump_2phases.py)) and 
three phases([`Salto_3phases.py`](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/Jump_3phases.py)) in order to learn how to use bioptim.

### Generate salto (3, 4, 5 and 6 phases)
The second step was to add a backward somersault during the flight phase, when the model attain the maximal height of 
his centre of mass (CoM). I begin by implement a backward somersault in a 3 phases movement ([`Salto_3phases.py`](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/Salto_3phases.py)) 
and gradually I increase the number of phases until reach 7 phases([`Salto_7phases.py`](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/Salto_7phases.py)).

### Actuator in dynamics
Add step actuator

### Robustness
After this, we want a code who produce two different techniques of backward somersault: 
one in 6 phases and an other with a waiting phase (i.e. phase 2) between the propulsion phase and the take-of phase.
These two methods have the same results for the preparation of propulsion and the propulsion phase.

![Robustness](/home/lim/Documents/Anais/Robust_standingBack/pictures/Dedoublement phase.png "Method for inducing robustness")


For all this steps, we list all the results of the simulation of jump and salto into a [Google Sheets](
https://docs.google.com/spreadsheets/d/1Zcdg7ftSXRW_HKXzb-tU153mgNU3cz4pQy1RCIJ5Snk/edit?usp=sharing).

## Model

The different model use are in the folder "Model". 
The name of the model is defines like this: `"Model2D_8Dof_2C_3M"` according to the type of the model (i.e. 2D or 3D), the number of degree of freedom (i.e. Dof) 
the number of contact (i.e. C) and the number of markers (i.e. M).

## How to save and visualize data

To save data, we use a function save_results_with_pickle, from the [Save file](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/Save.py), who use pickle to save all parameters (states, states dot, controls, cost, objectives, contraints)
To visualize data, we use the file [Visualisation](https://github.com/AnaisFarr/Robust_standingBack/blob/main/Code_examples/visualisation.py).


