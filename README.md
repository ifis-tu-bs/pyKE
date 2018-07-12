# pyKE

An Open-source library for Knowledge Embedding forked from [github.org/thunlp/OpenKE](http://github.org/thunlp/OpenKE).
The original API changed drastically to look more intuitively on a python notebook.


## Overview

This is an implementation based on [TensorFlow](http://www.tensorflow.org) for knowledge representation learning (KRL).
It includes native C++ implementations for underlying operations such as data preprocessing and negative sampling.
For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs.


## Installation

1. Clone repository and enter directory

    ```
    git clone https://github.com/ifis-tu-bs/pyKE.git
    cd pyKE
    ```

1. Install package

	`python setup.py install`


## Data

This framework requires datasets to contain a line with one number of elements followed by as many lines, each containing three whitespace-separated indices `head tail label` where `head` and `tail` denote indices of entities and `label` denotes the index of a relation.
Make sure to separate your data early on into at least two separate parts for training and testing.


## Quickstart

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, then train models and export results. For instance, we write an example_train_transe.py to train TransE:

	from pyke.dataset import Dataset
    from pyke.models import TransE
    
    # Read the dataset
    ds = Dataset("./benchmarks/fb15k.nt")
    
    # Configure parameters
    folds = 20
    neg_ent = 2
    neg_rel = 0
    
    
    # Set the knowledge embedding model class.
    def model():
        return TransE(50, 1.0, ds.ent_count, ds.rel_count, batch_size=ds.size // folds, variants=1 + neg_rel + neg_ent)
    
    
    # Train the model. It is saved in the process.
    model = ds.train(
        model,
        folds=folds,
        epochs=20,
        post_epoch=print,
        prefix="./TransE",
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=False,
        workers=4,
    )
    
    # Save the embedding to a JSON file
    model.save_to_json("TransE.json")


## Interfaces

### Dataset

`class pyke.dataset.Dataset` sets up the native library, handles the currently loaded dataset and defines the basic training algorithm.


### Base model class

`class pyke.models.BaseModel` declares the methods that all implemented model classes share, including the loss function neccessairy for training (inserting information into the model) and prediction (aka. retrieving information from the model).
This project implements the following model classes:

	class pyke.models.RESCAL
	class pyke.models.TransE
	class pyke.models.TransH
	class pyke.models.TransR
	class pyke.models.TransD
	class pyke.models.HolE
	class pyke.models.ComplEx
	class pyke.models.DistMult

