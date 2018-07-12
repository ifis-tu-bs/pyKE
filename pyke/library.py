import os
import subprocess
from ctypes import cdll, c_void_p, c_int64


class Library:
    """
    Manages the connection to the library.
    """
    library = None
    CPP_BASE = "cpp_library/Base.cpp"
    MAKE_SCRIPT = "cpp_library/make.sh"

    @staticmethod
    def compile_library(destination):
        pyke_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_base_file = os.path.join(pyke_dir, Library.CPP_BASE)
        make_script = os.path.join(pyke_dir, Library.MAKE_SCRIPT)
        call = [make_script, cpp_base_file, destination]
        subprocess.call(call)

    @staticmethod
    def load_library(path):
        lib = cdll.LoadLibrary(path)
        lib.sampling.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_int64]
        lib.bernSampling.argtypes = lib.sampling.argtypes
        lib.query_head.argtypes = [c_void_p, c_int64, c_int64]
        lib.query_tail.argtypes = [c_int64, c_void_p, c_int64]
        lib.query_rel.argtypes = [c_int64, c_int64, c_void_p]
        lib.importTrainFiles.argtypes = [c_void_p, c_int64, c_int64]
        lib.randReset.argtypes = [c_int64, c_int64]
        Library.library = lib

    @staticmethod
    def get_library(temp_dir=".pyke", library_name="pyke.so"):
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        library_path = os.path.abspath(os.path.join(temp_dir, library_name))

        if not os.path.exists(library_path):
            print("Compiling library ...")
            Library.compile_library(library_path)

        Library.load_library(library_path)

        return Library.library
