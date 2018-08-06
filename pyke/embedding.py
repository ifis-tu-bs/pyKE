import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd

from pyke import utils, norm, models
from pyke.dataset import Dataset
from pyke.library import Library

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("pyke")


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

    def predict_single(self, head_id, tail_id, rel_id):
        return self.__model.predict(head_id, tail_id, rel_id)

    def predict_multiple(self, head_ids, tail_ids, rel_ids):
        triples = zip(head_ids, tail_ids, rel_ids)
        return [self.predict_single(triple[0], triple[1], triple[2]) for triple in triples]

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
                             per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction,
                             learning_rate=self.learning_rate)

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

    def restore(self, prefix: str):
        # Create model
        model_parameters = self.get_model_parameters()
        self.__model = self.model_class(
            *model_parameters, ent_count=self.dataset.ent_count,
            rel_count=self.dataset.rel_count,
            batch_size=self.batch_size, variants=self.variants,
            optimizer=self.optimizer, norm_func=self.norm_func,
            per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction,
            learning_rate=self.learning_rate,
        )
        self.__model.restore(prefix)

    def meanrank(self, filtered=False, batch_count=None, head=True, tail=True, label=False, ):
        """
        Computes the mean rank of link prediction of one batch of the data.
        Returns floating values between 0 and size-1
        where lower results denote better models.
        The return value consists of up to three values,
        one for each of the columns 'head', 'tail' and 'label'.

            Arguments
        model - The to-be-tested embedding model.
        folds - Amount of batches the data is separated.
        index - Identifier of the tested batch.
        head, tail, label - Truth values denoting
        whether or not the respecting column should be tested.

            Note
        This test filters only 'false' facts evaluating better than the question.
        See `openke.meanrank` for the unfiltered, or 'raw', version.
        """
        # TODO: Only use training triples, currently only raw version
        if filtered:
            raise NotImplementedError("Filtered meanrank not implemented")
        if label:
            raise NotImplementedError("Meanrank for label not implemented")

        self.__library.randReset(self.workers, self.seed)
        sampling_func = self.__library.bernSampling if self.bern else self.__library.sampling
        types = (np.int64, np.int64, np.int64, np.float32)
        batches = [np.zeros(self.batch_size, dtype=t) for t in types]
        h_addr, t_addr, l_addr, y_addr = [utils.get_array_pointer(x) for x in batches]

        ranks = []
        folds = batch_count if batch_count else self.folds

        last_percent = 0.0
        limit = folds * self.batch_size
        print("Calculating mean rank ...")
        for _ in range(folds):
            sampling_func(h_addr, t_addr, l_addr, y_addr, self.batch_size, 0, 0, self.workers)
            for i in range(self.batch_size):
                h_id, t_id, r_id = batches[0][i], batches[1][i], batches[2][i]

                if head:
                    predictions = self.__model.predict(None, t_id, r_id)
                    df = pd.DataFrame(dict(
                        h=np.arange(self.dataset.ent_count),
                        t=np.full([self.dataset.ent_count], t_id),
                        r=np.full([self.dataset.ent_count], r_id),
                        y=predictions,
                    ))
                    df = df.sort_values("y")
                    df = df.reset_index(drop=True)
                    rank = df[df["h"] == h_id][df["t"] == t_id][df["r"] == r_id].index[0]
                    ranks.append(rank)
                if tail:
                    predictions = self.__model.predict(h_id, None, r_id)
                    df = pd.DataFrame(dict(
                        h=np.full([self.dataset.ent_count], h_id),
                        t=np.arange(self.dataset.ent_count),
                        r=np.full([self.dataset.ent_count], r_id),
                        y=predictions,
                    ))
                    df = df.sort_values("y")
                    df = df.reset_index(drop=True)
                    rank = df[df["h"] == h_id][df["t"] == t_id][df["r"] == r_id].index[0]
                    ranks.append(rank)

                percent = int((_ * self.batch_size + i) * 100.0 / limit)
                if percent > last_percent:
                    sys.stdout.write(f"\r{percent}% (i={i}, bs={self.batch_size}) ...")
                    sys.stdout.flush()
                    last_percent = percent

        sys.stdout.write(" done\n")
        return np.array(ranks).mean()
