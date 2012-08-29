#from distutils.core import setup
#from Cython.Build import cythonize

#setup(ext_modules=cythonize('*.pyx'))

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    ext_modules=[
        Extension("_malis",
                  sources=["_malis.pyx", "malis.cpp"],
                  language="c++",
                  include_dirs=[np.get_include()],
                 ),

    ],
    cmdclass={'build_ext': build_ext}
    )
