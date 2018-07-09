# ifisKE

An Open-source Framework for Knowledge Embedding forked from [github.org/thunlp/OpenKE](http://github.org/thunlp/OpenKE).
The original API changed drastically to look more intuitively on a python notebook.


## Overview

This is an implementation based on [TensorFlow](http://www.tensorflow.org) for knowledge representation learning (KRL).
It includes native C++ implementations for underlying operations such as data preprocessing and negative sampling.
For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs.


## Installation

1. Clone repository and enter directory

    ```
    git clone https://github.com/ifis-tu-bs/KnowledgeEmbedding.git
    cd KnowledgeEmbedding
    ```

1. Install requirements

	`pip install -r requirements.txt`

3. Build the library

	`./make.sh`


## Data

This framework requires datasets to contain a line with one number of elements followed by as many lines, each containing three whitespace-separated indices `head tail label` where `head` and `tail` denote indices of entities and `label` denotes the index of a relation.
Make sure to separate your data early on into at least two separate parts for training and testing.


## Quickstart

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, then train models and export results. For instance, we write an example_train_transe.py to train TransE:

	from openke import Dataset
	from openke.models import TransE

	#   Input training files from benchmarks/FB15K/ folder.
	with open("./benchmarks/FB15K/entity2id.txt") as f:
	    E = int(f.readline())
	with open("./benchmarks/FB15K/relation2id.txt") as f:
	    R = int(f.readline())

	#   Read the dataset.
	base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

	#   Set the knowledge embedding model class.
	model = TransE(50, 1.0, base.shape)

	#   Train the model.
	base.train(500, model, count=100, negatives=(1,0), bern=False, workers=4)

	#   Save the result.
	model.save("./result")


## Interfaces

### Config

`class openke.Dataset` in `openke/Config.py` sets up the native library, handles the currently loaded dataset and defines the basic training algorithm.


### Model Class

`class openke.models.ModelClass` in `openke/models/Base.py` declares the methods that all implemented model classes share, including the loss function neccessairy for training (inserting information into the model) and prediction (aka. retrieving information from the model).
This project implements the following model classes:

	class openke.models.RESCAL
	class openke.models.TransE
	class openke.models.TransH
	class openke.models.TransR
	class openke.models.TransD
	class openke.models.HolE
	class openke.models.ComplEx
	class openke.models.DistMult

