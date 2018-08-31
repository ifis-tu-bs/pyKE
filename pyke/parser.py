"""
Module contains different parsers to support multiple file types.

.. note:

   Currently only one file type is supported (N-Triples).
"""
import os
import sys
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from pyke.utils import split_nt_line, md5


class NTriplesParser:
    """
    Class creates benchmark files from a N-Triples file.
    """

    def __init__(self, filename: str, temp_dir: str, generate_valid_test: bool = False, fail_silently: bool = True):
        """
        Initializes the parser and creates the `temp_dir`.

        :param filename: Filename of the N-Triples file
        :param temp_dir: Directory where the benchmark should be placed
        :param generate_valid_test: Flag whether a validation and a test set should be created (default False)
        """
        self.filename = filename
        self.temp_dir = temp_dir
        self.generate_valid_test = generate_valid_test
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.file_hashsum = md5(filename)
        self.output_dir = os.path.join(self.temp_dir, self.file_hashsum)
        self.entity_file = os.path.join(self.output_dir, "entity2id.txt")
        self.relation_file = os.path.join(self.output_dir, "relation2id.txt")
        self.train_file = os.path.join(self.output_dir, "train2id.txt")
        self.valid_file = os.path.join(self.output_dir, "valid2id.txt")
        self.test_file = os.path.join(self.output_dir, "test2id.txt")
        self.ent_count = None
        self.rel_count = None
        self.train_count = None
        self.valid_count = None
        self.test_count = None
        self.fail_silently = fail_silently

    def parse(self):
        """
        Creates the benchmark files. Function checks if a benchmark exists for the specified file (using an MD5
        fingerprint) and loads this benchmark. Otherwise it creates a new benchmark.
        :return:
        """
        if os.path.exists(self.entity_file) and os.path.exists(self.relation_file) and os.path.exists(
                self.train_file) and (
                not self.generate_valid_test or (os.path.exists(self.valid_file) and os.path.exists(self.test_file))):
            print(f"Benchmark found: {self.output_dir}")
            with open(self.entity_file) as f:
                self.ent_count = int(f.readline())
            with open(self.relation_file) as f:
                self.rel_count = int(f.readline())
            with open(self.train_file) as f:
                self.train_count = int(f.readline())
            if self.generate_valid_test:
                with open(self.valid_file) as f:
                    self.valid_count = int(f.readline())
                with open(self.test_file) as f:
                    self.test_count = int(f.readline())
        else:
            self.create_benchmark()

    @staticmethod
    def read_triples(triples_fn: str) -> List[str]:
        """
        Returns the triple lines from the input files.

        :param triples_fn: Filename
        :return: List of lines from file
        """
        sys.stdout.write("Reading Triple lines ... ")
        sys.stdout.flush()
        lines = open(triples_fn, "r").readlines()
        print(str(len(lines)) + " Triple lines")
        return lines

    def map_triple_lines(self, triple_lines: List[str]):
        """
        Assigns each entity and each relation an id and creates a list of triples consisting of the ids.

        :param triple_lines: List of Triples
        :return: Tuple with the mapping of entity -> ID, relation -> ID and a list of the triples with the integer ids.
        """
        # prepare for mapping
        dict_ent = dict()
        dict_rel = dict()
        list_triples = []
        num_ent = 0
        num_rel = 0

        # iterate through every line containing a triple
        sys.stdout.write("Processing Triple lines ... ")
        sys.stdout.flush()
        progess = 0
        skipped = 0
        finish = len(triple_lines)
        last_percentage = 0
        for triple_line in triple_lines:
            # split it and check if it is a triple
            try:
                triple = split_nt_line(triple_line)
            except ValueError as e:
                if self.fail_silently:
                    skipped += 1
                    continue
                raise e

            # check if subject is in entities, add if not
            if triple[0] not in dict_ent:
                idx_sub = num_ent
                dict_ent.update({triple[0]: idx_sub})
                num_ent += 1
            else:
                idx_sub = dict_ent[triple[0]]

            # check if predicate is in relations, add if not
            if triple[1] not in dict_rel:
                idx_rel = num_rel
                dict_rel.update({triple[1]: idx_rel})
                num_rel += 1
            else:
                idx_rel = dict_rel[triple[1]]

            # check if object is in entities, add if not
            if triple[2] not in dict_ent:
                idx_obj = num_ent
                dict_ent.update({triple[2]: idx_obj})
                num_ent += 1
            else:
                idx_obj = dict_ent[triple[2]]

            # check triple
            if idx_sub < 0 or idx_rel < 0 or idx_obj < 0:
                sys.exit("Failure: Mapped Triple has invalid Indeces")

            # add to the mapped triples list
            # careful: OpenKE format is "subject object relation"
            mapped_triple = [idx_sub, idx_obj, idx_rel]
            list_triples.append(mapped_triple)

            # output progess (avoid spamming)
            progess += 1
            percentage = int((progess * 100) / finish)
            if percentage > last_percentage:
                sys.stdout.write("\rProcessing Triple lines ... " + str(percentage) + "%")
                sys.stdout.flush()
                last_percentage = percentage

        # output results
        print("")
        print(str(len(dict_ent)) + " Distinct Entities")
        print(str(len(dict_rel)) + " Distinct Relations")
        print(str(len(list_triples)) + " Distinct Triples")
        print("Skipped " + str(skipped) + " lines")
        return dict_ent, dict_rel, list_triples

    @staticmethod
    def convert_to_pandas(dict_ent: Dict[str, int], dict_rel: Dict[str, int], list_triples: List):
        """
        Converts to pandas series and dataframes

        :param dict_ent: Mapping entity to ID
        :param dict_rel: Mapping relation to ID
        :param list_triples: List with triples consisting of IDs
        :return: series of the entities, series of the relations and a dataframe of the triples with the IDs
        """
        print("Converting to Pandas Datastructure ...")
        srs_ent = pd.Series(list(dict_ent.keys()), index=list(dict_ent.values())).sort_index()
        srs_rel = pd.Series(list(dict_rel.keys()), index=list(dict_rel.values())).sort_index()
        df_triples = pd.DataFrame(list_triples)
        return srs_ent, srs_rel, df_triples

    @staticmethod
    def partition_data(df_triples: Any, generate_valid_test: bool = False, percentile_train: float = 0.8,
                       balance_valid_test: float = 0.5) -> Tuple[Any, Any, Any]:
        """
        Partitions triples dataframe to dataframes for training and (if specified) for validation and test

        :param DataFrame df_triples: DataFrame with the triples
        :param generate_valid_test: Flag indicating whether a validation and a test set should be created
        :param percentile_train:
        :param balance_valid_test:
        :return: Tuple with the training, validation and test set
        """
        print("Partitioning Data ...")
        if generate_valid_test:
            # random permutation of triples and partitioning in train, valid and test set
            print("Random permutation and partition of Triples in Training-, Validation- and Test Data ...")
            df_triples_len = len(df_triples.index)
            df_triples = df_triples.reindex(np.random.permutation(df_triples_len).tolist())
            size_train = int(df_triples_len * percentile_train)
            size_valid = int((df_triples_len - size_train) * balance_valid_test)
            size_test = df_triples_len - size_train - size_valid
            df_train = df_triples.head(size_train).sort_index()
            df_valid = df_triples.tail(size_valid + size_test).head(size_valid).sort_index()
            df_test = df_triples.tail(size_test).sort_index()
            return df_train, df_valid, df_test
        else:
            # no validation and test set, all triples are a training set
            print("No Validation- and Test Data, only Training Data with all Triples.")
            df_train = df_triples.sort_index()
            return df_train, pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def save_element2id(series: pd.Series, file_name: str = "element2id.txt"):
        """
        Saves elements (entities / relations)

        :param series: Series of the elements (entities/relations)
        :param file_name: Target file
        """
        sys.stdout.write("Saving Elements to %s ... " % file_name)
        sys.stdout.flush()
        file_elements = open(file_name, "w")
        file_elements.write(str(len(series.index)))
        for idx, element in series.iteritems():
            file_elements.write("\n" + str(element) + "\t" + str(idx))
        file_elements.close()
        print("Done")

    @staticmethod
    def save_triple2id(df_triples: Any, file_name: str = "triple2id.txt"):
        """
        Saves triple data

        :param DataFrame df_triples: DataFrame with triples
        :param file_name: Target file
        """
        sys.stdout.write("Saving Triples to %s ... " % file_name)
        sys.stdout.flush()
        file_triples = open(file_name, "w")
        file_triples.write(str(len(df_triples.index)))
        for triple in df_triples.itertuples():
            file_triples.write("\n" + str(triple[1])
                               + "\t" + str(triple[2])
                               + "\t" + str(triple[3]))
        file_triples.close()
        print("Done")

    def create_benchmark(self):
        """
        Creates the benchmark from the filename given in the constructor
        """
        # read triples
        triple_lines = self.read_triples(self.filename)

        # map triple lines
        dict_ent, dict_rel, list_triples = self.map_triple_lines(triple_lines)

        # convert to pandas series and dataframe
        srs_ent, srs_rel, df_triples = self.convert_to_pandas(dict_ent, dict_rel, list_triples)

        # partition data
        df_train, df_valid, df_test = self.partition_data(df_triples, self.generate_valid_test, 0.8, 0.5)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.save_element2id(srs_ent, self.entity_file)
        self.save_element2id(srs_rel, self.relation_file)

        # save training data and (if specified) validation and test data
        self.save_triple2id(df_train, self.train_file)
        if self.generate_valid_test:
            self.save_triple2id(df_valid, self.valid_file)
            self.save_triple2id(df_test, self.test_file)

        self.ent_count = len(srs_ent)
        self.rel_count = len(srs_rel)
        self.train_count = len(df_train)
        self.test_count = len(df_test)
        self.valid_count = len(df_valid)
