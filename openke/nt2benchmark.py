# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd

# opens files and reads contents from isql output (ignore first 6 rows and last 2 rows)
from openke.utils import split_nt_line


def read_triples(triples_fn):
    sys.stdout.write("Reading Triple lines ... ")
    sys.stdout.flush()
    lines = open(triples_fn, "r").readlines()
    print(str(len(lines)) + " Triple lines")
    return lines


# map triple lines to dictionaries and triple list
def map_triple_lines(triple_lines):
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
        triple = split_nt_line(triple_line)

        if len(triple) != 3:
            sys.exit("Failure: Line is not a valid N-Triples line (parsing \"" + triple_line + "\")")

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


# converts to pandas series and dataframes
def convert_to_pandas(dict_ent, dict_rel, list_triples):
    print("Converting to Pandas Datastructure ...")
    srs_ent = pd.Series(list(dict_ent.keys()), index=list(dict_ent.values())).sort_index()
    srs_rel = pd.Series(list(dict_rel.keys()), index=list(dict_rel.values())).sort_index()
    df_triples = pd.DataFrame(list_triples)
    return srs_ent, srs_rel, df_triples


# partitions triples dataframe to dataframes for training and (if specified) for validation and test
def partition_data(df_triples, generate_valid_test=False, percentile_train=0.8, balance_valid_test=0.5):
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
        return df_train, None, None


# saves elements (entities / relations)
def save_element2id(series, file_name="element2id.txt"):
    sys.stdout.write("Saving Elements to %s ... " % file_name)
    sys.stdout.flush()
    file_elements = open(file_name, "w")
    file_elements.write(str(len(series.index)))
    for idx, element in series.iteritems():
        file_elements.write("\n" + str(element) + "\t" + str(idx))
    file_elements.close()
    print("Done")


# saves triple data
def save_triple2id(df_triples, file_name="triple2id.txt"):
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


def nt2benchmark(filename, output_dir, generate_valid_test):
    # read triples
    triple_lines = read_triples(filename)

    # map triple lines
    dict_ent, dict_rel, list_triples = map_triple_lines(triple_lines)

    # convert to pandas series and dataframe
    srs_ent, srs_rel, df_triples = convert_to_pandas(dict_ent, dict_rel, list_triples)

    # partition data
    df_train, df_valid, df_test = partition_data(df_triples, generate_valid_test, 0.8, 0.5)

    # save entities and relations
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_element2id(srs_ent, os.path.join(output_dir, "entity2id.txt"))
    save_element2id(srs_rel, os.path.join(output_dir, "relation2id.txt"))

    # save training data and (if specified) validation and test data
    save_triple2id(df_train, os.path.join(output_dir, "train2id.txt"))
    if generate_valid_test:
        save_triple2id(df_valid, os.path.join(output_dir, "valid2id.txt"))
        save_triple2id(df_test, os.path.join(output_dir, "test2id.txt"))

    return len(srs_ent), len(srs_rel)
