from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


setup(
    ext_modules=[
        Extension("_demo",
                  sources=["_demo.pyx", "demo.cpp"],
                  language="c++",
                  include_dirs=[np.get_include()],
                 ),

    ],
    cmdclass={'build_ext': build_ext}
    )


import _demo

arr = np.random.randn(10).astype('f')
print _demo.func(arr, size=len(arr))
