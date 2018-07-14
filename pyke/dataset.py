# -*- coding: utf-8 -*-
import ctypes

import numpy as np

from pyke.library import Library
from pyke.parser import NTriplesParser
from pyke.utils import get_array_pointer


class Dataset(object):
    """
    Manages a collection of relational data
    encoded as a set of triples, each describing a statement (or fact)
    over two objects called 'entities', one 'head' and one 'tail',
    being related in a manner that is symbolized by a relation 'label'.

    The application encodes both entities and relations as integral values,
    describing an index in an ordered table.
    """

    def __init__(self, filename: str, temp_dir: str = ".pyke"):
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
        :param temp_dir: Directory for storing the benchmark files. Application needs write access
        """
        self.__library = Library.get_library(temp_dir)

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

    def query(self, head, tail, relation):
        """
        Checks which facts are stored in the entire dataset.
        This method is overloaded for the task of link prediction,
        awaiting an incomplete statement and returning all known substitutes.

        :param head: Index of a head entity.
        :param tail: Index of a tail entity.
        :param relation: Index of a relation label.

        :return: A boolean array, deciding for each candidate whether or not the resulting statement is
            contained in the dataset.
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
