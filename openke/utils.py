# -*- coding: utf-8 -*-
import hashlib


def split_nt_line(line):
    """
    Splits a line from a N-triples file into subject, predicate and object.

    :param str line: Line from a N-triples file
    :return: tuple with subject, predicate, object
    """
    s, p, o = line.split(sep=" ", maxsplit=2)
    s = s.lstrip("<").rstrip(">")
    p = p.lstrip("<").rstrip(">")
    o = o.strip().rstrip(" .")
    if o.startswith("<"):
        o = o.lstrip("<").rstrip(">")
    return s, p, o


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
