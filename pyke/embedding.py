import datetime as dt
import os

import numpy as np

from pyke import utils
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
        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def batch_size(self):
        return self.dataset.size // self.folds

    @property
    def batch_size_neg(self):
        return self.batch_size * (1 + self.neg_ent + self.neg_rel)

    # TODO: Replace model constructor with self.model_class
    # TODO: Modify models to accept kwargs in the constructor
    # TODO: Add cross validation
    def train(self, model_constructor, folds=1, epochs=1, model_count=1, prefix='best', post_epoch=None,
              post_batch=None, autosave: float = 30, continue_training=True, **kwargs):
        """
        Training algorithm over the whole dataset. The model is saved as a TensorFlow binary at the end of the training
        and every `autosave` minutes.

        :param model_constructor: Parameterless constructor of to-be-trained embedding models
        :param folds: Number of batches
        :param epochs: Number of epochs
        :param model_count: Number of models to be trained. The best model is selected
        :param prefix: Prefix to save the model
        :param post_epoch: Callback at the end of each epoch (receiving epoch number and loss)
        :param post_batch: Callback at the end of each batch (receiving batches and loss)
        :param autosave: Time in minute after which the model is saved.
        :param continue_training: Flag, which states whether the training process should continue with the existing
            model. If False, the model is newly trained.
        :param kwargs: Optional kwargs for the batch creation. Possible values are: neg_ent, neg_rel, bern, workers,
            seed
        """
        self.__library.randReset(self.workers, self.seed)
        sampling_func = self.__library.bernSampling if self.bern else self.__library.sampling
        types = (np.int64, np.int64, np.int64, np.float32)
        batches = [np.zeros(self.batch_size_neg, dtype=t) for t in types]
        h_addr, t_addr, l_addr, y_addr = [utils.get_array_pointer(x) for x in batches]

        # create model
        m = model_constructor()
        # m = self.model_class(**
        if os.path.exists(prefix + ".index") and continue_training:
            print(f"Found model with prefix {prefix}. Continuing training ...")
            m.restore(prefix)
        datetime_next_save = dt.datetime.now() + dt.timedelta(minutes=autosave)

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

        m.save(prefix)
        self.__model = m

    def save_to_json(self, path: str):
        self.__model.save_to_json(path)
