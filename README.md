# RDDRS
This repository contains code and trained models for the paper "Riemannian data-dependent randomized smoothing for neural network certification" by Pol Labarbarie, Hatem Hajri and Marc Arnaudon.

<p>
<img src="figures/merge_2D_data.png" height="448" width="896" >
</p>

## How does it work?

This repo is based on previous works and then on previous codes. You may find bellow a list of the associate repositories.

* ["Randomized Smoothing"](https://github.com/locuslab/smoothing)
* ["Data Dependent Randomized Smoothing"](https://github.com/MotasemAlfarra/Data_Dependent_Randomized_Smoothing)
* ["ANCER: Anisotropic Certification via Sample-wise Volume Maximization"](https://github.com/MotasemAlfarra/ANCER)

### Environment Installations:
First, you may prefer to create a virtual environment by running for example for conda: 

`conda create -n myenv python==3.9`

Then, activate the envionment by running:

`conda activate myenv`

Now you can install the requirments packages by running:

`pip install -r requirements.txt`

### Scripts

* In the program [main.py](code/main.py), you may choose if you want to perform a training or directrly use a trained model and then you may choose what certification method you want use:

```python code/main.py ```  

* In the program [riemannian.py](code/riemannian.py) you may find the optimization stage that we develop.

## Citation

If you use this repo, please cite us.
