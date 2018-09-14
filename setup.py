from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extension = [
        Extension('procrutes',
            ['src/procrutes.pyx'],
            language='c++',
            include_dirs = [numpy.get_include()])]

setup(ext_modules = cythonize(extension))
