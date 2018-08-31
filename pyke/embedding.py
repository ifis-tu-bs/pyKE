import datetime
import logging
import os
import sys

import numpy as np

from pyke import models
from pyke.dataset import Dataset
from pyke.library import Library
from pyke.openke import Config
from pyke.utils import get_rank

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("pyke")


class Embedding:
    """
    Class is a Wrapper for an embedding. It consists of a dataset and a model and provides an interface
    for the normal embedding operations such as prediction, training, saving and restoring.

    :param optimizer: Possible values: SGD, Adagrad, Adadelta, Adam
    """

    def __init__(self, dataset: Dataset, model_class: type, **kwargs):
        self.dataset = dataset
        self.model_class = model_class
        # self.__model = None
        self.__config = None
        self.__library = Library.get_library()
        # Training args
        self.neg_ent = 1
        self.neg_rel = 0
        self.bern = True
        self.workers = 1
        self.folds = 1
        self.epochs = 1
        self.optimizer = "SGD"
        self.per_process_gpu_memory_fraction = 0.5
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
        self.__init_config()

    def __str__(self):
        return f"<Embedding: {self.model_class.__name__.split('.')[-1]} {self.get_model_parameters()}>"

    def __init_config(self):
        """Wrapper for the config object"""
        con = Config()
        con.set_in_path(self.dataset.benchmark_dir)
        con.set_test_link_prediction(self.dataset.generate_valid_test)
        con.set_test_triple_classification(self.dataset.generate_valid_test)
        con.set_work_threads(self.workers)
        con.set_train_times(self.epochs)
        con.set_nbatches(self.folds)
        con.set_alpha(self.learning_rate)
        con.set_lmbda(self.weight)
        con.set_margin(self.margin)
        con.set_bern(int(self.bern))
        con.set_ent_dimension(self.ent_dim)
        con.set_rel_dimension(self.rel_dim)
        con.set_dimension(self.dimension)
        con.set_ent_neg_rate(self.neg_ent)
        con.set_rel_neg_rate(self.neg_rel)
        con.set_opt_method(self.optimizer)
        con.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
        con.init()
        con.set_model(self.model_class)
        self.__config = con

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
            raise ValueError(f"Model class {self.model_class.__name__} is not supported.")

    @property
    def batch_size(self):
        return self.dataset.size // self.folds

    @property
    def batch_size_neg(self):
        return self.batch_size * (1 + self.neg_ent + self.neg_rel)

    @property
    def variants(self):
        return self.neg_rel + self.neg_ent + 1

    def predict(self, head_id, tail_id, rel_id):
        heads = [head_id] if isinstance(head_id, int) else head_id
        tails = [tail_id] if isinstance(tail_id, int) else tail_id
        rels = [rel_id] if isinstance(rel_id, int) else rel_id

        if head_id is None:
            if tail_id is None:
                if rel_id is None:
                    raise NotImplementedError('universal prediction')
                raise NotImplementedError('full-relation prediction')
            elif rel_id is None:
                raise NotImplementedError('full-tail prediction')
            heads, tails, rels = np.arange(self.dataset.ent_count), np.full([self.dataset.ent_count], tail_id), \
                                 np.full([self.dataset.ent_count], rel_id)
        elif tail_id is None:
            if rel_id is None:
                raise NotImplementedError('full-head prediction')
            heads, tails, rels = np.full([self.dataset.ent_count], head_id), np.arange(self.dataset.ent_count), \
                                 np.full([self.dataset.ent_count], rel_id)
        elif rel_id is None:
            heads, tails, rels = np.full([self.dataset.rel_count], head_id), np.full([self.dataset.rel_count], tail_id), \
                                 np.arange(self.dataset.rel_count)

        if isinstance(head_id, int) and isinstance(tail_id, int) and isinstance(rel_id, int):
            return self.__config.test_step(heads, tails, rels)[0]
        return self.__config.test_step(heads, tails, rels)

    def train(self, prefix='best', save_steps: int = 100, continue_training=True):
        """
        Train the embedding.

        :param prefix: Model prefix to save
        :param save_steps: Steps after which the model is saved
        :param continue_training: If true and an existing model is found, the training is resumed
        """
        if os.path.exists(prefix + ".index") and continue_training:
            print(f"Found model with prefix {prefix}. Continuing training ...")
            self.restore(prefix)
        else:
            self.__config.set_import_files(None)
        self.__config.set_export_files(prefix, save_steps)
        self.__config.run()  # TODO: Add cross validation

    def save_to_json(self, path: str):
        """
        Save embedding to JSON.

        :param path: JSON path
        """
        self.__config.save_parameters(path)

    def restore(self, prefix: str):
        """
        Loads an existing embedding.

        :param prefix: Prefix of the model files
        """
        self.__config.set_import_files(prefix)
        self.__config.restore_tensorflow()

    def get_validation_triples(self):
        """
        Returns a list of triples used for the metrics.
        """
        return self.dataset.valid_set if self.dataset.generate_valid_test else self.dataset.train_set

    def meanrank(self, filtered=False, head=True, tail=True, label=False):
        """
        Computes the mean rank of the embedding.
        """
        if filtered:
            raise NotImplementedError("Filtered meanrank not implemented")

        ranks = []
        triples = self.get_validation_triples()
        last_percent = 0.0
        count = len(triples)

        start_time = datetime.datetime.now()
        sys.stdout.write(f"Calculating mean rank ...")
        for idx, (head_id, tail_id, label_id) in enumerate(triples):
            value = self.predict(head_id, tail_id, label_id)
            if head:
                predictions = self.predict(None, tail_id, label_id)
                rank = get_rank(predictions, value)
                ranks.append(rank)
            if tail:
                predictions = self.predict(head_id, None, label_id)
                rank = get_rank(predictions, value)
                ranks.append(rank)
            if label:
                predictions = self.predict(head_id, tail_id, None)
                rank = get_rank(predictions, value)
                ranks.append(rank)

            percent = idx * 100.0 / count
            if percent > last_percent:
                sys.stdout.write(f"\rCalculating mean rank ... {percent:.2f} %")
                sys.stdout.flush()
                last_percent = percent

        sys.stdout.write(f"\rCalculating mean rank ... done in {datetime.datetime.now() - start_time}\n")
        return np.array(ranks).mean()

    def hits_at_k(self, k: int, filtered: bool = False):
        """
        Calculates the hits@k metric (raw or filtered) for the embedding.

        :param k: First top k elements to look at
        :param filtered: flat for filtered hits@k (otherwise raw)
        """
        raise NotImplementedError("Hits@k is currently not implemented.")

    def get_ent_embeddings(self):
        """
        Returns the entity embedding.

        :return: Entity embedding as numpy matrix
        """
        return self.__config.get_parameters_by_name("ent_embeddings")

    def get_parameters(self):
        """
        Returns all embedding parameters in dependence of the model These can be the entity embedding, relation
        embedding, transfer matrices, etc.

        :return: dictionary with parameters
        """
        return self.__config.get_parameters()
