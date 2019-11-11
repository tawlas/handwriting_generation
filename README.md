# Handwriting generation!

Author: Alassane Watt at [JRL-Lab](http://jrl-umi3218.github.io/) at AIST Tsukuba Research (Japan) and CentraleSupelec University (https://www.centralesupelec.fr/) (France).

### Introduction

Each model and its dependencies are gathered in a specific folder (conditional_generation and unconditional_generation)
In each of those folders, you will find a specs.json file that contains the parameters of the corresponding model.

Warning: You may need to modify path files in the "specs.json" files of the models since they are set for launching in notebooks (results in notebook folder) which are not in the root of this project.

### Training the random genration model

1. run on CLI:

    ```Shell
	python3 train_ugm.py -e unconditionl_generation
	```

	To see the plotting of the loss (during or after training time), run :

	 ```Shell
	python3 plot_log.py -e unconditionl_generation
	```

### Generating random stroke

1.  Once the model is trained and saved in its folder (see specs.json), run:

	```Shell
	python3 ugm_inference.py
	```
    Potentially give an argument for random seed. For example:

    ```Shell
	python3 ugm_inference.py -r 10
	```
    A plot will pop up drawing the stroke.

### Training the conditional generation model

1. run on CLI:

    ```Shell
	python3 train_cgm.py -e conditionl_generation
	```

	To see the plotting of the loss (during or after training time), run :

	 ```Shell
	python3 plot_log.py -e conditionl_generation
	```

### Generating conditioned stroke

1.  Once the model is trained and saved in its folder (see specs.json), run for example:

	```Shell
	python3 cgm_inference.py -s "Hello World"
	```
    -s argument is the sentence to condition on.

    A plot will pop up drawing the stroke.
