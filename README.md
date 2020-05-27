# Hashing Models

This repository includes the adversarial autoencoder models for the following papers:

* [Image Hashing by Minimizing Discrete Component-wise Wasserstein Distance (arxiv'20)](https://arxiv.org/abs/2003.00134)
* [Efficient Implicit Unsupervised Text Hashing using Adversarial Autoencoder (WWW'20)](https://dl.acm.org/doi/abs/10.1145/3366423.3380150)
* [Adversarial Factorization Autoencoder for Look-alike Modeling (CIKM'19)](https://dl.acm.org/doi/abs/10.1145/3357384.3357807)

Please cite the appropriate paper(s) if you are using this repository. If you have any questions, please also feel free to send us an email at khoadoan@vt.edu.

## What does this reposistory has?

This repository includes the following features:

* Code to create the common experimental evaluations for the learning-to-hash task,
* Various implemented ranking metrics for the learning-to-hash task.
* Visualization code.
* Several neural net models that were used in the original papers. *Note: due to time constraint, we will add/update the models gradually*
* Adversarial training algorithm for learning the hash functions.

## 1. Setup

### Prerequisites

We need the following:

* conda or miniconda (preferred)
* GPU or CPU
* your perseverance :smiley:

### Install the environment

Clone the repository and run the following commands inside the basedir of this repository:

```
conda create --name hashing --file requirements.txt python=3.7
conda activate hashing
```

and you are ready to have some fun. Your perseverance stops here! Wait, more like "your perseverance starts here" :grin:!!!

## To run the jupyer notebook

You can start the jupyter notebook server by invoking the following command:

```
CUDA_VISIBLE_DEVICES=<gpu ids> bin/run_notebook <id>
```

This will open a notebook server at port 888<id> on your machine.

If you don't have a GPU, you can run the notebook on the CPU by the following command.

```
bin/run_notebook_cpu
```

This will open a notebook server at port 8888 on your machine if it's available.

## Repository artifacts

* `python`: all the python code will be here
* `notebooks`: all the notebooks will be here
* `bin`: all the executables will be here
* `data`: data will be here (but it's not checked in)
* `code`: all the external code references will be here (e.g. external dependent repos)
* `requirements.txt`: list of python reqs
* `README.md`: this doc, and light documentation of this repos.

## 2. Create the experiment data

The notebook `notebooks/create_datasets.ipynb` contains the following tasks:

* Randomly split the datasets into **train**, **db**, **query** sets. There are different sizes of the splits, each of which is specific to some papers in the literature.
* The creations of the following datasets do not need any manual downloads: mnist, fashion-mnist, cifar10. These datasets are downloaded automatically using `torchvision`.
* The creation of the following datasets need manual downloads into the `data` folder before we can split them: flickr25k, place365.

*Note: for flickr25k, download the dataset at this [link](https://bit.ly/2TDLKjc).*

To extract the **VGG-pretrained** features, use the notebook `extract_VGG_features.ipynb`. We can extract the features at different intermediate layers of the VGG network.

## 3. Run Aversarial Autoencoderss

The experiments on each dataset will be in the notebook `notebooks/ae-<dataset>.ipynb`. 
