# NBV-Net:  A 3D Convolutional Neural Network for Predicting the Next-Best-View

This repository is a modified and cleaned version of the original code for Next Best View network proposed by Mendoza in the following preprint:
> Mendoza, M., Vasquez-Gomez, J. I., Taud, H., Sucar, L. E., & Reta, C. (2019). Supervised Learning of the Next-Best-View for 3D Object Reconstruction. arXiv preprint arXiv:1905.05833.

The modifications are authored by Alexey I. Boyko and Dmitry Smirnov as a part of Foundations of DS class at Skoltech.

## Try simplified version (non-packaged) on Colab
[PyTorch Colab](https://colab.research.google.com/github/aiboyko/nbv-net/blob/master/NBW_net.ipynb)


## Getting Started with the package
### Installation
- Clone this repo:
```bash
git clone https://github.com/aiboyko/nbv-net
cd nbv-net
```
- Build and install
```bash
pip install .
```

### Hardware requirements
Due to a relatively large dataset and networks the workstation should have at least 12GB RAM and 6Gb GRAM

## Dataset
The [original dataset](https://www.kaggle.com/miguelmg/nbv-dataset) can be downloaded from Kaggle
Extract it to ```/dataset``` and run ```scripts/dataset_preparation.py```
  
The package has ```__main__.py``` file, so it can be invoked as a script by 'python -m cnbv'
