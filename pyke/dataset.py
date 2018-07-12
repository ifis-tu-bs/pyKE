# -*- coding: utf-8 -*-
import ctypes
import datetime as dt
import os

import numpy as np

from pyke.library import Library
from pyke.parser import NTriplesParser


class Dataset(object):
    """
    Manages a collection of relational data
    encoded as a set of triples, each describing a statement (or fact)
    over two objects called 'entities', one 'head' and one 'tail',
    being related in a manner that is symbolized by a relation 'label'.

    The application encodes both entities and relations as integral values,
    describing an index in an ordered table.
    """

    def __init__(self, filename: str, library: str = './libopenke.so', temp_dir: str = ".pyke"):
        """
        Creates a new dataset from a N-triples file.

        .. note:

           The N-triples file is parsed into the original OpenKE benchmark file structure containing a file for the
           entities (entity2id.txt), for the relations (relation2id.txt) and the training file (train2id.txt). These
           files are stored by default in the `.pyke` directory in a subdirectory named after the MD5-sum of the
           input file. The MD5-sum is used to prevent the tool from recreating the benchmark files for.
           If you change the N-triples file the MD5-sum changes and so the entities and relations get a new id.

        .. note:

           At the moment, no two datasets can be open at the same time!

        :param filename: Pathname to the N-triples file for training
        :param library: Path to a shared object implementing the preprocessing
        :param temp_dir: Directory for storing the benchmark files. Application needs write access
        """
        global _l
        self.__library = _l[library]

        parser = NTriplesParser(filename, temp_dir)
        parser.parse()

        self.__library.importTrainFiles(
            ctypes.c_char_p(bytes(parser.train_file, 'utf-8')),
            parser.ent_count,
            parser.rel_count,
        )
        self.size = parser.train_count
        self.ent_count = parser.ent_count
        self.rel_count = parser.rel_count
        self.shape = self.ent_count, self.rel_count

    def __len__(self):
        """Returns the size of the dataset."""
        return self.size

    # FIXME: Function produces strange behaviour (see #7)
    # def get_batches(self, count, neg_ent=0, neg_rel=0, bern=True, workers=1, seed=1):
    #     """
    #     Separates the dataset into nearly equal parts.
    #     Iterates over all parts, each time yielding four arrays.
    #     This method can be used for cross-validation as shown below:
    #
    #         assert numpy, base, folds, epochs, Model
    #
    #         for i in range(folds):
    #
    #             model = Model()
    #
    #             for _ in range(epochs):
    #                 for j, b in enumerate(base.batches(folds)):
    #                     if i != j:
    #                         model.fit(*b)
    #
    #             for j, b in enumerate(base.batches(folds)):
    #                 if i == j:
    #                     break
    #             score = model.predict(*b[:3])
    #     """
    #     batch_size = self.size // count
    #     batch_size_neg = batch_size * (1 + neg_ent + neg_rel)
    #     types = (np.int64, np.int64, np.int64, np.float32)
    #     batches = [np.zeros(batch_size_neg, dtype=t) for t in types]
    #     h_addr, t_addr, l_addr, y_addr = [get_array_pointer(x) for x in batches]
    # TODO: Move randReset
    #     self.__library.randReset(workers, seed)
    #
    #     sampling = self.__library.bernSampling if bern else self.__library.sampling
    #     for _ in range(count):
    #         sampling(h_addr, t_addr, l_addr, y_addr, batch_size, neg_ent, neg_rel, workers)
    #         yield batches

    # TODO: Add selection for best-performing model
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
        # Prepare batches
        neg_ent = kwargs.get("neg_ent", 1)
        neg_rel = kwargs.get("neg_rel", 0)
        bern = kwargs.get("bern", True)
        workers = kwargs.get("workers", 1)
        seed = kwargs.get("seed", 1)
        batch_size = self.size // folds
        batch_size_neg = batch_size * (1 + neg_ent + neg_rel)
        self.__library.randReset(workers, seed)
        sampling_func = self.__library.bernSampling if bern else self.__library.sampling
        types = (np.int64, np.int64, np.int64, np.float32)
        batches = [np.zeros(batch_size_neg, dtype=t) for t in types]
        h_addr, t_addr, l_addr, y_addr = [get_array_pointer(x) for x in batches]

        # create model
        m = model_constructor()
        if os.path.exists(prefix + ".index") and continue_training:
            print(f"Found model with prefix {prefix}. Continuing training ...")
            m.restore(prefix)
        datetime_next_save = dt.datetime.now() + dt.timedelta(minutes=autosave)

        for epoch in range(epochs):
            loss = 0.0
            for _ in range(folds):
                # create batch
                sampling_func(h_addr, t_addr, l_addr, y_addr, batch_size, neg_ent, neg_rel, workers)

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
        return m

    # FIXME: Function currently uses get_batches(), which produces strange behaviour (see #7)
    # def meanrank(self, model, folds=1, index=0, head=True, tail=True, label=True, batchkwargs={}):
    #     """
    #     Computes the mean rank of link prediction of one batch of the data.
    #     Returns floating values between 0 and size-1
    #     where lower results denote better models.
    #     The return value consists of up to three values,
    #     one for each of the columns 'head', 'tail' and 'label'.
    #
    #         Arguments
    #     model - The to-be-tested embedding model.
    #     folds - Amount of batches the data is separated.
    #     index - Identifier of the tested batch.
    #     head, tail, label - Truth values denoting
    #     whether or not the respecting column should be tested.
    #
    #         Note
    #     This test filters only 'false' facts evaluating better than the question.
    #     See `openke.meanrank` for the unfiltered, or 'raw', version.
    #     """
    #
    #     def rank(d, x, h, t, l):
    #         y = self.query(h, t, l)
    #         z = model.predict(h, t, l)
    #         return sum(1 for i in range(self.shape[d]) if z[i] < z[x] and not y[i])
    #
    #     I = lambda: range(self.size // folds)
    #     for i, (h, t, l, _) in enumerate(self.get_batches(folds, **batchkwargs)):  # TODO: change index variable
    #         if i == index:
    #             break
    #     ranks = [
    #         (rank(0, h[i], None, t[i], l[i]) for i in I()) if head else None,
    #         (rank(0, t[i], h[i], None, l[i]) for i in I()) if tail else None,
    #         (rank(1, l[i], h[i], t[i], None) for i in I()) if label else None,
    #     ]
    #
    #     return [sum(i) / self.size for i in ranks if i is not None]

    def query(self, head, tail, relation):
        """
        Checks which facts are stored in the entire dataset.
        This method is overloaded for the task of link prediction,
        awaiting an incomplete statement and returning all known substitutes.

            Arguments
        head - Index of a head entity.
        tail - Index of a tail entity.
        label - Index of a relation label.

            Return Value
        A boolean array, deciding for each candidate
        whether or not the resulting statement is contained in the dataset.
        """
        if head is None:
            if tail is None:
                if relation is None:
                    raise NotImplementedError('querying everything')
                raise NotImplementedError('querying full relation')
            if relation is None:
                raise NotImplementedError('querying full head')
            heads = np.zeros(self.shape[0], np.bool_)
            self.__library.query_head(get_array_pointer(heads), tail, relation)
            return heads
        if tail is None:
            if relation is None:
                raise NotImplementedError('querying full tail')
            tails = np.zeros(self.shape[0], np.bool_)
            self.__library.query_tail(head, get_array_pointer(tails), relation)
            return tails
        if relation is None:
            relations = np.zeros(self.shape[1], np.bool_)
            self.__library.query_rel(head, tail, get_array_pointer(relations))
            return relations
        raise NotImplementedError('querying single facts')


def get_array_pointer(a):
    """
    Returns the address of the numpy array.

    :param a: Numpy array
    :return: Memory address of the array
    """
    return a.__array_interface__['data'][0]


_l = Library()
