# OpenKE
An Open-source Framework for Knowledge Embedding.
This project was forked from [github.org/thunlp/OpenKE](http://github.org/thunlp/OpenKE).

## Overview
This is an efficient implementation based on [TensorFlow](http://www.tensorflow.org) for knowledge representation learning (KRL).
We use C++ to implement some underlying operations such as data preprocessing and negative sampling.
For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs.

## Installation

1. Install requirements

	$ pip install tensorflow

2. Clone the OpenKE repository:

	$ git clone https://github.com/thunlp/OpenKE

	$ cd OpenKE

3. Build the library

	$ bash make.sh

## Data

This framework requires datasets to contain a line with one number of elements followed by as many lines, each containing three whitespace-separated indices `head tail label` where `head` and `tail` denote indices of entities and `label` denotes the index of a relation.
Make sure to separate your data early on into at least two separate parts for training and testing.

## Quickstart

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, then train models and export results. For instance, we write an example_train_transe.py to train TransE:


	from openke import Config
	from openke.models import TransE

	con = Config()

	#   Input training files from benchmarks/FB15K/ folder.
	with open("./benchmarks/FB15K/entity2id.txt") as f:
	    E = int(f.readline())
	with open("./benchmarks/FB15K/relation2id.txt") as f:
	    R = int(f.readline())
	con.init("./benchmarks/FB15K/train2id.txt", E, R, batch_count=100, negative_entities=1)

	#   Models will be exported via tf.Saver() automatically.
	con.set_export("./res/model.vec.tf", None, 1)

	#   Set the knowledge embedding model
	from tensorflow.python.training.gradient_descent import GradientDescentOptimizer as SGD
	con.set_model(TransE, SGD(.001), hidden_size=50, margin=1.0)
	#   Train the model.
	con.train(500, bern=False, workers=4)

## Interfaces

### Config

`class openke.Config.Config` sets up the native library, handles the currently loaded dataset and embedding model and defines the basic training algorithm.

### Model Class

`class openke.models.Base.ModelClass` declares the methods that all implemented model classes share, including the loss function neccessairy for training (inserting information into the model) and prediction (aka. retrieving information from the model).
This project implements the following model classes:

	class openke.models.RESCAL.RESCAL
	class openke.models.TransE.TransE
	class openke.models.TransH.TransH
	class openke.models.TransR.TransR
	class openke.models.TransD.TransD
	class openke.models.HolE.HolE
	class openke.models.ComplEx.ComplEx
	class openke.models.DistMult.DistMult

