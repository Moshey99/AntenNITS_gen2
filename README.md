# AntenNITS
Antenna design based on Neural Inverse Transform Sampler (https://proceedings.mlr.press/v162/li22j).

This project is a solution to the problem of designing antennas with specific spec.
The project uses conditional version of NITS to generate samples from the prior distribution of the antenna parameters that match the desired spec.

# Environment Setup
Using conda is recommended for setting up the environment.
1. Create a new conda environment with 3.11 python version, using the following command:\
for example, let's name the environment "antenna":\
``` conda create --name antenna python=3.11 ```
2. Activate the environment:\
``` conda activate antenna ```
3. cd to the project directory:\
``` cd /PATH/TO/PROJECT/AntenNITS_gen2 ```
4. Install the required packages:\
``` pip install -r requirements.txt ``` 

# Data
The project expects the data to be in the following format:

|-- processed_data_130k_200k/
-  |-- 130000/
   - |-- antenna.npy
   - |-- environment.npy
   - |-- gamma.npy
   - |-- radiation.npy
   - |-- radiation_directivity.npy
     
-  |-- 130001/
   - ... same as above
-  |-- EXAMPLE/
   - |-- ant_parameters.pickle
   - |-- model_parameters.pickle
-  |-- ant_scaler.pkl
-  |-- env_scaler.pkl
-  |-- checkpoints/
   - |-- forward_best_dict.pth
-  |-- checkpoints_inverse/
   - |-- ANT_model_lr_0.0002_hd_512_nr_8_pd_0.9_bs_12_drp_0.31_bounds_-3_3.pth

## Data Description
1. **Data folders (130000,130001,...)** - contain the data. Each sample contains:
   1. Description of the antenna in antenna.npy
   2. The environment of the antenna in environment.npy
   3. Radiation (of the antenna in the environment) in terms of gain (in dB) in radiation.npy
   4. Radiation in terms of directivity (in dB) in Radiation_directivity.npy
   5. S parameter (in dB) in gamma.npy
   * **note**: Both radiation and S parameter are represented as a concatenation of the magnitude and then the phase.
2. **ant_scaler.pkl & env_scaler.pkl** - metadata to standardize the antenna and environment representation, respectively.
See details on how they are generated in the **Usage** section.
3. **checkpoints** folder - The path where forward model's weights are saved. Contains forward_best_dict.pth
which is a pre-trained weights for the model.
4. **checkpoints_inverse** folder - The path where inverse model's weights are saved. Contains ANT_model_lr_0.0002_hd_512_nr_8_pd_0.9_bs_12_drp_0.31_bounds_-3_3.pth
pre-trained weights for the inverse model.\
Moreover, this is the folder where generated antennas are saved into by default.

# Repository Description
This section breaks down the repository with some explanation about the files.
## AntennaDesign/
An upgraded version milestone 2. Main responsibility of this folder is:
- Define the forward model.
- Define the loss functions for the forward model.
- ```utils.py```, which is the super important file, which is responsible for:
   - data loading for training and evaluation
   - statistics 
   - original model/CST interface utility functions (validity function, representation conversion, etc.)
   - raw data processing
   - regular utility functions (linear to dB conversion for S parameter and radiation, vice versa, etc.)
### EXAMPLE/
An auxiliary folder for ```utils.py``` in order to deal with the original representation of the model, as
defined in the CST model interface.
### models/
The architecture of the forward model and its submodules. Main forward module is in ```forward_GammaRad.py```.
### losses.py
All the losses for the forward model. Main forward model loss is expressed in the class ```GammaRad_loss```.
### utils.py
described above.
## nits
All the modules for NITS, including ```antenna_condition.py``` which is part of the modification for conditioning 
the spectrum on the network.
## notebooks
Here we have all the main scripts, which are detailed in the Usage section

# Usage
All the scripts that needs to be executed are in **notebooks** folder in this project.\
Each script contains an argument parser with explanation over the arguments.

## process_data.py
Takes the original data, as it is provided from the data generator (CST) and returns the processed data,
represented with numpy. processing the antenna and environment data automatically generates ant_scaler and env_scaler.
## forward_model_train_main.py
Trains the forward model. Saves in ```checkpoints``` the best weights of the model,
as well as temporary weights (after every multiple epochs)
## forward_model_evaluate_main.py
Evaluates the forward model. Enables to plot visual result of the forward model, and also generate statistics for the error.
## inverse_model_train_main.py
Trains the inverse model. Saves in ```checkpoints_inverse``` best weights of the model,
as well as temporary weights (after every multiple epochs)
## inverse_model_generate_samples_main.py
Generates antennas given a trained inverse model and a dataset. It saves the samples in a
numpy format into ```checkpoints_inverse/samples``` by default.
## inverse_model_evaluate_main.py
Uses the samples generated from inver_model_generate_samples_main script in order to evaluate visually and statistically (using the forward model) 
the  generated samples. Moreover, it transforms the samples from numpy to the format that specified in the simulation,
which allows to evaluate the samples with exact simulator, to produce exact evaluation later on.


# General Important Notes:
- It is recommended to work with PyCharm.
- Mark ```AntennaDesign``` folder as Source root to use it as a starting point for resolving imports.\
That is done in PyCharm by right-click on the folder, then **Mark Directory as --> Sources root**
- The current project loads gain for the radiation, but it can be easily modified in ```AntennaDataSet.load_antenna``` method
in ```utils.py``` to directivity by loading the appropriate file.