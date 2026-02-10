# TII_SGlow
Seeking Safety from Uncertainty: Probabilistic Vessel Trajectory Prediction with a Flow-based Generative Model.


## Code Statement

This repository contains the official implementation, all comparison experiments, and ablation studies from the paper. The project is structured into several key directories and modules for clarity.

### Main Framework
The core code for model training, evaluation, and prediction is located in the root directory.
- `main_run.py` -- The main entrance and execution script. Handles parameter configuration and outputs batch predictions.
- `dataloader.py` -- Loads and preprocesses the AIS data, including normalization, dataset splitting, and batch preparation for prediction.
- `processor.py` -- The main operational framework, responsible for loading/saving models, and executing training, testing, and prediction procedures.
- `utils.py` -- Contains utility functions, primarily for calculating evaluation metrics.
- `visualization.py` -- Provides scripts for visualizing a batch of predicted trajectories.

### Data Preprocessing (`AIS_process/`)
This module handles the conversion of raw AIS data into a format suitable for model training.
- `AIS_process.py` -- The main script to run. Processes raw AIS data into training samples. Each batch contains multiple samples, representing approximately 120 ships and their corresponding adjacency matrices.
- `Functions.py` -- Contains auxiliary functions called by the main preprocessing script.

### Models (`Models/`)
This directory contains the implementations of our proposed model (**SGlow**), several comparative deep learning models, and the ablation study variants.
- **Proposed Model:**
    - `SGlow.py` -- The core implementation of our main model, **SGlow**.
- **Comparative Models (Complex Implementations):**
    - `RealNVP.py`, `Glow.py`, `GAN.py`, `VAE.py`, `CVAE.py`, `STGCN.py`
- **Comparative Models (Simple Implementations):**
    - `Compared_models.py` -- Implementations of several standard sequence models: LSTM, GRU, TCN, Seq2Seq, and Transformer.
- **Foundation for Flow Models:**
    - `Flows.py` -- Contains base layers and modules (e.g., affine coupling layer, invertible 1x1 convolution, loss computation) that support the implementations of **SGlow**, Glow, and RealNVP.
- **Ablation Studies:**
    - `Ablas.py` -- Implementation code for the ablation experiments, featuring multiple ablated variants of the **SGlow** model.


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


