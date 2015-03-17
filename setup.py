# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

numpy_include_dir = np.get_include()

patch_module = Extension(
    "patch_utils._patch_utils",
    ["_patch_utils.pyx"],
    language="c++",
    include_dirs=[
        numpy_include_dir
    ]
)

setup(
    name="patch_utils",
    version="0.1",
    description="Fast extraction of patches in n-dimensional images",
    author="Pablo Márquez Neila",
    author_email="pablo.marquezneila@epfl.ch",
    package_dir={'patch_utils': ''},
    packages=[
        "patch_utils",
        "patch_utils.scripts"
    ],
    ext_modules=cythonize([patch_module]),
    requires=['numpy', 'Cython']
)
