from ctypes import cdll, c_void_p, c_int64


class Library:
    """
    Manages the connection to the library.
    """
    library_name = ""

    def getLibrary(self):
        pass

    def __init__(self):
        self.__dict = dict()

    def __getitem__(self, key):
        if key in self.__dict:
            return self.__dict[key]
        lib = cdll.LoadLibrary(key)
        lib.sampling.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_int64]
        lib.bernSampling.argtypes = lib.sampling.argtypes
        lib.query_head.argtypes = [c_void_p, c_int64, c_int64]
        lib.query_tail.argtypes = [c_int64, c_void_p, c_int64]
        lib.query_rel.argtypes = [c_int64, c_int64, c_void_p]
        lib.importTrainFiles.argtypes = [c_void_p, c_int64, c_int64]
        lib.randReset.argtypes = [c_int64, c_int64]
        self.__dict[key] = lib
        return lib
