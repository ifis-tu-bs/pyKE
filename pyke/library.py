import os
import subprocess
from ctypes import cdll, c_void_p, c_int64


class Library:
    """
    Manages the connection to the library.

    .. note:

       This library should be converted to a python module in the future.
    """
    library = None
    temp_dir = ".pyke"
    library_name = "pyke.so"
    CPP_BASE = "cpp_library/Base.cpp"
    MAKE_SCRIPT = "cpp_library/make.sh"

    @staticmethod
    def compile_library(destination: str):
        """
        Compile the library to the path ``destination``.

        :param destination: path for the library
        """
        pyke_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_base_file = os.path.join(pyke_dir, Library.CPP_BASE)
        make_script = os.path.join(pyke_dir, Library.MAKE_SCRIPT)
        call = [make_script, cpp_base_file, destination]
        subprocess.call(call)

    @staticmethod
    def load_library(path: str):
        """
        Loads the library from `path`.

        :param path: path to the library (.so)
        """
        lib = cdll.LoadLibrary(path)
        lib.sampling.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64]
        lib.getHeadBatch.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.getTailBatch.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.testHead.argtypes = [c_void_p]
        lib.testTail.argtypes = [c_void_p]
        lib.getTestBatch.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        lib.getValidBatch.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        lib.getBestThreshold.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.test_triple_classification.argtypes = [c_void_p, c_void_p, c_void_p]
        Library.library = lib

    @staticmethod
    def get_library(temp_dir: str = None, library_name: str = None):
        """
        Return the C++ library. The function compiles it if it doesn't exist and it loads the library.

        :param temp_dir: directory where the library is saved (optional)
        :param library_name: filename of the library
        :return: c++ library
        """
        if temp_dir:
            Library.temp_dir = temp_dir
        if library_name:
            Library.library_name = library_name

        if not os.path.exists(Library.temp_dir):
            os.mkdir(Library.temp_dir)

        library_path = os.path.abspath(os.path.join(Library.temp_dir, Library.library_name))

        if not os.path.exists(library_path):
            print("Compiling library ...")
            Library.compile_library(library_path)

        if not Library.library:
            Library.load_library(library_path)

        return Library.library
