from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "zfit_wrapper",
        sources=["zfit_wrapper.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=["."],
        libraries=["zfit", "gomp"],
        extra_link_args=["-Wl,-rpath,."],
    )
]

setup(
    name="zfit_wrapper",
    ext_modules=cythonize(ext_modules, language_level="3"),
)
