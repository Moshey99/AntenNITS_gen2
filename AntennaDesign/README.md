# AntennaDesign
Code for Antenna Design
## Contents:
- losses.py - various loss function to train/evaluate different models.
- models - forward/inverse/forward & inverse concatenated model architectures.
- utils.py - different utilities like nearest neighbor benchmark, data preprocessing, etc. 
- trainer.py - training and evaluating functions.
- antenna_training.py - main script. loads data,preprocess it, defines hyper-parameters and runs it all.

* Important: if inverse model is hypernetwork, then batch size MUST be 1. Learning process can be modified by
gradient accumulation parameter in the main script. If inverse is not hypernetwork, then batch size can be larger and no gradient accumulation is needed.

## checkpoints:
- INVERSE_GammaRad_concat_HuberCyclic_loss_lamda1_radphasefac1_lr0.0002_loss0.674.pth - trained inverse_forward model 
that predicts both gamma and radiation pattern.
- FORWARD_small_10layers_dB_linpred.pth - trained forward model that predicts gamma linearly.
- FORWARD_radiation_HuberCyclic_loss_range[-55,5].pth - trained forward model that predicts radiation pattern [dB].
## models:
 - baseline_regressor.small_deeper_baseline_forward_model - architecture of forward model for gamma.  
 - forward_radiation.Radiation_Generator - architecture of forward model for radiation pattern.
 - forward_GammaRad.forward_GammaRad - architecture of forward model for both gamma and radiation pattern.
 - inverse_hypernet.small_inverse_radiation_no_hyper - inverse model
 - inverse_hypernet.inverse_forward_concat - inverse and forward concatenated
## How to train:
  - #### training models is done in antenna_training.py . The code receives multiple arguments (data path, hyper-parameters, etc.) and trains model's weights (and saves them, if wanted). In addition, this script plots the value of the loss function VS epochs
    1. Define the required arguments in arg_parser as you wish (by changing the value in "default") or leave them as they are.
       <img width="572" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/a2a107e4-85e0-440a-a326-e8e3c4432484">
    2. Define manually model's architecture and the loss function it will try to minimize. There are various models and loss functions, so recommended for a start to do that:
       ```
           model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                                    forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range),
                                                    forward_weights_path_rad=args.forward_model_path_radiation,
                                                    forward_weights_path_gamma=args.forward_model_path_gamma)
           loss_fn = GammaRad_loss(lamda=GammaRad_lambda,rad_phase_fac=rad_phase_fac)
       ```
       that defines the model to be a inverse-forward concatenation with loaded&frozen forward weights, and the loss function to be GammaRad_loss (takes into account both radiation pattern and gamma)
    - **IMPORTANT**: the --inv_or_forw argument must match the chosen architecure. In the case above it must be ``` inverse_forward_GammaRad ```.
      for all the options look into the function create_dataloader in utils.py . 
    3. In the end of the file there is a line of code that saves model weights in "checkpoints" folder ( torch.save(model.state_dict(), ...) ).
      <img width="555" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/c0fd299a-3f48-4daf-ac7b-700b3b7b1af3">

      - if saving the weights is not needed (for example, you do not want to use this model for inference), put this line in comment.
      - if you want to use it for inference, please choose the name for the file with .pth extension, for example 'checkpoints/INVERSE_GammaRad_concat_HuberCyclic_loss.pth' . the weights will be saved there.
    4.  Run the file.

## How to evaluate a model:
 - #### evaluating (in inference time) a model is done in inverse_forward_GammaRad_eval.py ( It is guarenteed to work for inverse-forward concat model that predicts both gamma and radiation. Other models might require slight changes).
   the script takes gets the weights of the model (as well as data_path, inv_or_forw, etc. Please see arg_parser in this script), and computes several things:
   a. statistics for the predictions of radiation and gamma (avg and max error, and ms-ssim for radiation) on **all the examples** in the validation/test set.
      
      example:
      <img width="880" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/5d3fbd9f-07f4-49a0-8ea9-6064cb5f1afb">
      
   **note**: The image is cropped, there should be on the right the value of the MS-SSIM metric in addition.
   b. the prediction of the spectrum of a given sample from the validation/test set, together with the GT spectrum
      
      example:
      ![image](https://github.com/Moshey99/AntennaDesign/assets/104683567/6e0019c3-d4e0-4320-bda5-9a8a76e1d719)
   c. The predicted geometry, together with the GT geometry.
      
      example:
      <img width="518" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/a248731a-fd5a-4284-9b76-3e6334916a72">

1. Please choose for --model_path the path of the learned weights you with to evaluate. It can be the path of the weight you trained by your own, or pre-trained weights (such as INVERSE_GammaRad_concat_HuberCyclic_loss_lamda1_radphasefac1_lr0.0002_loss0.674.pth that is described in the checkpoints section).
2. choose --sample (integer between 0 and 1920), --data_path, etc. (look into arg_parser).
3. Run the file.

## plot/output details:
A typical output of **inverse_forward_GammaRad_eval.py** will look like that:
![image](https://github.com/Moshey99/AntennaDesign/assets/104683567/5b4db931-03a6-4fa9-bc01-63086fa75a45)

- The very right plot evaluates GT vs predicted gamma coefficient. Each is of the size 502, represented as a concatenation of magnitude and then phase, for 251 sampled frequencies. The black dotted line separates visually magnitude values and phase values
- The lower plots evaluates GT (right) vs predicted (left) radiation pattern's magnitude of a certain sample. Each is of the size 46x46, meaning each represents a downsampled angular space.
- The upper plots evaluates GT (right) vs predicted (left) radiation pattern's phase of a certain sample. Each is of the size 46x46, meaning each represents a downsampled angular space.
  
In addition, as mentioned in the evaluation section, the code prints:
- Results of different statistics/metrics that help for model evaluation and investigation
- Predicted geometry, that the model finds as the best fit for the spectral input. the geometry represented by a simple 12 elements vector, where each is a dof of the model. The meaning of each element is as following: <img width="327" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/31e03fc3-5370-48f0-8e93-fb04b3a0db77"> . There are 9 lengths (3 metals, with 3 changable lengths), width, excitation coordinate dof, and one more dof.
  
These vectors define the geometrical configuration of the metals. The next image shows general geometry, and the difference when 'l12' set to two different values (1 and 5).

<img width="578" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/0db7d24a-f315-4b9f-9289-5a41013c2776">


      


