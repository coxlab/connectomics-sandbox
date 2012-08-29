# distutils: language = c++
# distutils: sources = malis.cpp

cdef extern from "malis.h" namespace "malis":
    cdef int cppmalis(int)

import numpy as np
cimport numpy as cnp

def test(
    cnp.ndarray[cnp.float_t, ndim=1] arr,
    x=1,
    ):
    ptr = arr.data
    return cppmalis(x)
