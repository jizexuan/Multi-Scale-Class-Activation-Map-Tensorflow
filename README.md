# Multi-Scale Class Activation Map (MS-CAM)
This repository contains tensorflow implemenation of MS-CAM for geographic atrophy (GA) lesions segmentation:
* We perform a weakly supervised training strategy with image level labels.
* We construct a multi-scale neural network to generate multi-scale class activation maps.
* We apply this model to SD-OCT data with GA. The final segmentation results are displayed on the en face projection images.


## Contents
- [Dataset](#Dataset)
- [Note](#Note)
	- [Requirements](#Requirements)
   	- [Running](#Running)
- [Architecture](#Architecture)

## Dataset
We utlize two different GA datasets to evaluate the model. Each dataset contains SD-OCT cubes with three dimensions. All the cubes contain the advanced non-exudative AMD with GA.
> Our datasets were acquired with a Cirrus OCT device from Stanford University.
* Each OCT cube-sacn should be split into B-scans as the input of network.
* Each OCT cube-sacn has an en face manual ground truth. For weakly supervised training phase, 
* Some pre-processing steps can be done on softwares, e.g. OCTExplorer.

## Note

* [`train_test.py`](train_test.py): training and testing on SD-OCT datasets, with cross validation
* [`network.py`](network.py): network construction, with modules of generating class activation maps
* [`utils.py`](utils.py): required functions
* [`demo.ipynb`](demo.ipynb): a demo of lesion segmentation over the trained network
### * Requirements
* My running environments, based on Ubuntu 18.04:
    * CUDA==10.2
    * CUDNN==7.6.5
    * Python==3.7
    * TensorFlow==1.14.0
    
* One-line installation
```sh
pip install -r requirements.txt
```

### * Running
* Make sure all flags are filled in correctly, and the data is placed in the specified path. Run the following command to run the cross validation.
```sh
python train_test.py
```

## Architecture

### * To Do
* [ ] Publish the address of our paper
* [ ] Introduce the architecture of the network
