README
======

An Open-source library for Knowledge Embedding forked from
`github.org/thunlp/OpenKE <http://github.org/thunlp/OpenKE>`_.
The original API changed drastically to be more pythonic.


Overview
--------

This is an implementation based on [TensorFlow](http://www.tensorflow.org) for knowledge representation learning (KRL).
It includes native C++ implementations for underlying operations such as data preprocessing and negative sampling.
For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient 
platform to run models on GPUs.


Installation
------------

1. Clone repository and enter directory

   .. code::

      git clone https://github.com/ifis-tu-bs/pyKE.git
      cd pyKE

2. Install package

   .. code::

      python setup.py install


Quickstart
----------

To compute a knowledge graph embedding, first import datasets and set configure parameters for training, 
then train models and export results. Here is an example to train the FB15K dataset with the TransE model.

.. code::

   from pyke.dataset import Dataset
   from pyke.embedding import Embedding
   from pyke.models import TransE
    
   # Read the dataset
   dataset = Dataset("./benchmarks/fb15k.nt")
   embedding = Embedding(
       dataset,
       TransE,
       folds=20,
       epochs=20,
       neg_ent=1,
       neg_rel=0,
       bern=False,
       workers=4,
       dimension=50,  # TransE-specific
       margin=1.0,  # TransE-specific
   )

   # Train the model. It is saved in the process.
   embedding.train(prefix="./TransE", post_epoch=print)

   # Save the embedding to a JSON file
   embedding.save_to_json("TransE.json")


Interfaces
----------

The class `pyke.embedding.Embedding` represents an embedding which requires a dataset and a model class.
Initialize your data set in form of a N-triples file with the class `pyke.dataset.Dataset`.


Models
......

The class `pyke.models.base.BaseModel` declares the methods that all implemented model classes share, including the loss function neccessairy for training (inserting information into the model) and prediction (aka. retrieving information from the model).
This project implements the following model classes:

- RESCAL
- TransE
- TransH
- TransR
- TransD
- HolE
- ComplEx
- DistMult

Notes
-----

The original fork consists of a C++ library which is compiled once you use the project. Please note,
that the compilation is only supported on **UNIX-based systems**. In the future the C++ library should be replaced
by a python library.
