import datetime as dt
import os

import numpy as np

from pyke import utils, norm, models
from pyke.dataset import Dataset
from pyke.library import Library


class Embedding:
    """
    Class is a Wrapper for an embedding. It consists of a dataset and a model and provides an interface
    for the normal embedding operations such as prediction, training, saving and restoring.
    """

    def __init__(self, dataset: Dataset, model_class: type, **kwargs):
        self.dataset = dataset
        self.model_class = model_class
        self.__model = None
        self.__library = Library.get_library()
        # Training args
        self.neg_ent = 1
        self.neg_rel = 0
        self.bern = True
        self.workers = 1
        self.seed = 1
        self.folds = 1
        self.epochs = 1
        self.optimizer = None
        self.per_process_gpu_memory_fraction = 0.5
        self.norm_func = norm.l1
        self.learning_rate = 0.01
        # Model specific parameters
        self.dimension = 50  # ComplEx, DistMult, HolE, RESCAL, TransD, TransE, TransH
        self.ent_dim = 50  # TransR
        self.rel_dim = 10  # TransR
        self.margin = 1.0  # HolE, RESCAL, TransD, TransE, TransH, TransR
        self.weight = 0.0001  # ComplEx, DistMult
        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_model_parameters(self):
        """
        Returns the model-specific parameters used by the constructor. These are for example dimension, weight, margin.
        The tuple can be unpacked with *args*.
        Returning

        - (dimension, weight) for ComplEx and DistMult,
        - (ent_dim, rel_dim, margin) for TransR,
        - (dimension, margin) for HolE, RESCAL, TransE, TransD, TransH.

        :return: tuple with the model specific parameters
        """
        if self.model_class in (models.ComplEx, models.DistMult):
            return self.dimension, self.weight
        elif self.model_class == models.TransR:
            return self.ent_dim, self.rel_dim, self.margin
        elif self.model_class in (models.HolE, models.RESCAL, models.TransE, models.TransD, models.TransH):
            return self.dimension, self.margin
        else:
            raise ValueError(f"Model class {self.model_class} is not supported.")

    @property
    def batch_size(self):
        return self.dataset.size // self.folds

    @property
    def batch_size_neg(self):
        return self.batch_size * (1 + self.neg_ent + self.neg_rel)

    @property
    def variants(self):
        return self.neg_rel + self.neg_ent + 1

    # TODO: Replace model constructor with self.model_class
    # TODO: Modify models to accept kwargs in the constructor
    # TODO: Add cross validation
    def train(self, prefix='best', post_epoch=None, post_batch=None, autosave: float = 30, continue_training=True):
        """
        Training algorithm over the whole dataset. The model is saved as a TensorFlow binary at the end of the training
        and every `autosave` minutes.

        :param prefix: Prefix to save the model
        :param post_epoch: Callback at the end of each epoch (receiving epoch number and loss)
        :param post_batch: Callback at the end of each batch (receiving batches and loss)
        :param autosave: Time in minute after which the model is saved.
        :param continue_training: Flag, which states whether the training process should continue with the existing
            model. If False, the model is newly trained.
        """
        # Initialize batches
        self.__library.randReset(self.workers, self.seed)
        sampling_func = self.__library.bernSampling if self.bern else self.__library.sampling
        types = (np.int64, np.int64, np.int64, np.float32)
        batches = [np.zeros(self.batch_size_neg, dtype=t) for t in types]
        h_addr, t_addr, l_addr, y_addr = [utils.get_array_pointer(x) for x in batches]

        # Create model
        model_parameters = self.get_model_parameters()
        m = self.model_class(*model_parameters, ent_count=self.dataset.ent_count, rel_count=self.dataset.rel_count,
                             batch_size=self.batch_size, variants=self.variants,
                             optimizer=self.optimizer, norm_func=self.norm_func,
                             per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)

        # Load existing model
        if os.path.exists(prefix + ".index") and continue_training:
            print(f"Found model with prefix {prefix}. Continuing training ...")
            m.restore(prefix)
        datetime_next_save = dt.datetime.now() + dt.timedelta(minutes=autosave)

        # Training process
        for epoch in range(self.epochs):
            loss = 0.0
            for _ in range(self.folds):
                # create batch
                sampling_func(h_addr, t_addr, l_addr, y_addr, self.batch_size, self.neg_ent, self.neg_rel, self.workers)

                # train step
                loss += m.fit(*batches)

                # post-batch callback
                if post_batch:
                    post_batch(batches, loss)

            # post-epoch callback
            if post_epoch:
                post_epoch(epoch, loss)

            # Save
            if dt.datetime.now() > datetime_next_save:
                print(f"Autosave in epoch {epoch} ...")
                m.save(prefix)
                datetime_next_save = dt.datetime.now() + dt.timedelta(minutes=autosave)

        # Save model
        m.save(prefix)
        self.__model = m

    def save_to_json(self, path: str):
        """
        Save embedding to JSON.

        :param path: JSON path
        """
        self.__model.save_to_json(path)
