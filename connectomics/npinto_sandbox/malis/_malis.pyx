# distutils: language = c++
# distutils: sources = malis.cpp

cdef extern from "malis.h" namespace "malis":
    cdef int cppmalis(int)

def test(x=1):
    return cppmalis(x)
