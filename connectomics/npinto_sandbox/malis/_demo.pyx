# distutils: language = c++
# distutils: sources = demo.cpp

cdef extern from "demo.h" namespace "demo":
    cdef int test(float*, int)

import numpy as np
cimport numpy as cnp

def func(cnp.ndarray[cnp.float32_t, ndim=1] arr,
         size=1,
        ):

    assert arr.dtype == 'float32'
    ptr = arr.data
    return test(<float*>arr.data, <Py_ssize_t>size)
