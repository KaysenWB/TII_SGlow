# TII_SGlow
Seeking Safety from Uncertainty: Probabilistic Vessel Trajectory Prediction with a Flow-based Generative Model.


## Code Statement

This repository contains the official implementation, all comparison experiments, and ablation studies from the paper. The project is structured into several key directories and modules for clarity.

### Main Framework
The core code for model training, evaluation, and prediction is located in the root directory.
- `main_run.py` -- The main entrance and execution script. Handles parameter configuration and outputs predictions.
- `dataloader.py` -- Loads and preprocesses the AIS data, including normalization, dataset splitting, and batch preparation for prediction.
- `processor.py` -- The main operational framework, responsible for loading/saving models, and executing training, testing, and prediction procedures.
- `utils.py` -- Contains utility functions, primarily for calculating evaluation metrics.
- `visualization.py` -- Provides scripts for visualizing a batch of predicted trajectories.

### AIS_process Folder
This directory handles the conversion of raw AIS data into a format suitable for model training.
- `AIS_process.py` -- The main script to run. Processes raw AIS data into training samples. Each batch contains multiple samples, representing approximately 120 ships and their corresponding adjacency matrices.
- `Functions.py` -- Contains auxiliary functions called by the main preprocessing script.

### Models Folder
This directory contains the implementations of our proposed model (SGlow), several comparative deep learning models, and the ablation study variants.
- `SGlow.py` -- Code for the implementation of the main model SGlow. 
- `RealNVP.py` & `Glow.py` & `GAN.py` & `VAE.py` & `CVAE.py` & `STGCN.py`  -- Code for several complex implementations of comparative models.
- `Compared_models.py`  -- Code for several simple implementations of comparative models, including LSTM, GRU, TCN, Seq2Seq,Transformer.
- `Flows.py` -- Layers and modules of the flow base models, including affine coupling layer, invertible convolutional layer and loss computation. They are used to support implementations of SGlow and its ablated variants, Glow and RealNVP.
- `Ablas.py` -- Implementation code for ablation experiments, multiple ablation variants of SGlow.


## Environment Setup

**System Requirements**

- Operating System: Linux (Ubuntu 18.04+ recommended)
- Python 3.8 or higher
- CUDA 11.3+ (for GPU acceleration, optional)

**Dependencies**

- torch==2.8.0
- numpy==2.0.1
- pandas==2.3.3
- math==1.3.0
- pytorch_tcn==1.2.3
- matplotlib == 3.7.2
- nflows == 0.14


## Network Structure

Figure01: Network structure of SGlow.
![Figure01](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig2.jpg?raw=true)

Figure02: Network structure a flow including AFF and InvConv.

<div align="center">
  <img src="https://github.com/KaysenWB/TII_SGlow/blob/main/Fig3.jpg?raw=true" 
       alt="Figure02" 
       width="50%" />
</div>

## Results
Figure03: Qualitative comparison results of 12 deep learning models, including six discriminative models and five generative models.
![Figure03](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig5.jpg?raw=true)

Figure04: SGlow's qualitative findings in different encounters.
![Figure04](https://github.com/KaysenWB/TII_SGlow/blob/main/Fig6.jpg?raw=true)



## Citation
If you find this repository useful in your research, please consider citing the following papers:
```
@ARTICLE{11534643,
  author={Yang, Kaisen and Lu, Yuxu and Yang, Dong},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Seeking Safety From Uncertainty: Probabilistic Vessel Trajectory Prediction With a Flow-Based Generative Model}, 
  year={2026},
  pages={1-11},
  publisher={IEEE}
```

## Contact
If you have any queries or are interested in developing collaborations and communications, please contact me via email: kaisen.yang@connect.polyu.hk (Yang Kaisen).

