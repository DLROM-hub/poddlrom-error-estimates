# Error estimates for POD-DL-ROMs
This repository contains the official source code implementation of the paper 

Brivio, S., Fresca, S., Franco, N.R., Manzoni, A., Error estimates for POD-DL-ROMs: a deep learning framework for reduced order modeling of nonlinear parametrized PDEs enhanced by proper orthogonal decomposition. Adv Comput Math 50, 33 (2024). https://doi.org/10.1007/s10444-024-10110-1. 

### Installation
We suggest to install the repository in a new conda environment, namely,
```
conda create -n poddlrom python=3.10
conda activate poddlrom
conda install nvidia::cuda-nvcc=11.8
pip install -r requirements.txt --no-cache-dir
```

### Instructions
In the folder ```tests``` we provide a sample numerical experiment to showcase a possible usage
of the implemented library.

### Cite
If the present repository and/or the original paper was useful in your research, 
please consider citing

```
@article{brivio2024error,
title={Error estimates for {P}{O}{D}-{D}{L}-{R}{O}{M}s: a deep learning framework for reduced order modeling of nonlinear parametrized {P}{D}{E}s enhanced by proper orthogonal decomposition}, 
author={Simone Brivio and Stefania Fresca and Nicola Rares Franco and Andrea Manzoni},
year={2024},
journal = {Adv. Comput. Math.},
volume = {50}, 
number = {33}
}
```

### Data availability
We provide the dataset for the available numerical experiment ```ibvp2d``` at [this link](https://drive.google.com/drive/u/1/folders/1l-OsrWcEuJ_da6FqDhXjuXPjXundIxg7).
