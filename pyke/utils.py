# -*- coding: utf-8 -*-
import hashlib


def split_nt_line(line: str):
    """
    Splits a line from a N-triples file into subject, predicate and object.

    :param line: Line from a N-triples file
    :return: tuple with subject, predicate, object
    """
    s, p, o = line.split(maxsplit=2)
    s = s.lstrip("<").rstrip(">")
    p = p.lstrip("<").rstrip(">")
    o = o.strip().rstrip(" .")
    if o.startswith("<"):
        o = o.lstrip("<").rstrip(">")
    return s, p, o


def md5(filename: str):
    """
    Returns the MD5-hashsum of a file.

    :param filename: Filename
    :return: MD5-hashsum of the file
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_array_pointer(a):
    """
    Returns the address of the numpy array.

    :param a: Numpy array
    :return: Memory address of the array
    """
    return a.__array_interface__['data'][0]