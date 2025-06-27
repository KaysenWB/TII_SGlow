# TII_SGlow
Seeking Safety from Uncertainty: Probabilistic Vessel Trajectory Prediction with a Flow-based Generative Model.

# Code Statement
This code contains all the comparison and ablation experiments from the paper. Below are descriptions of the each file:

main_run.py -- Main entrance and run files for the model, including parameter settings and output of a batch of predictions.  
dataloader.py -- Loading the processed AIS data, and some more detailed processing before entering, such as normalisation, dividing the dataset, and setting up the batches to be predicted.  
processor.py -- A framework for model operation, including loading the network, saving the network, training, testing and prediction operations.  
utils.py -- Some additional functions.  
visualization.py -- Visualising a batch of predicted trajectories.  




AIS_process: AIS data preprocessing, AIS_process.py is the running file and Functions.py is the functions to be called.



# Network Structure

Figure01
Network structure of SGlow.
![Figure01](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig2.jpg?raw=true)

Figure02
Network structure a flow including AFF and InvConv.

<div align="center">
  <img src="https://github.com/KaysenWB/TII_SGlow/blob/main/Fig3.jpg?raw=true" 
       alt="Figure02" 
       width="50%" />
</div>

# Results
Figure03
Qualitative comparison results of 12 deep learning models, including six discriminative models and five generative models.
![Figure03](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig5.jpg?raw=true)

Figure04
SGlow's qualitative findings in different encounters.
![Figure04](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig6.jpg?raw=true)


