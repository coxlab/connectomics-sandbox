import time
import numpy as np


def theano_memory_hack(func_exp, local_vars, 
                       input_exp='input', slice_exp='slice',
                       msize_start=512, msize_factor=1.8,
                       verbose=False):
    """Super-hacky way of computing theano expression on large datasets.

    Just the first dimension will be "sliced". This is especially useful
    to get around memory errors while using the GPU.

    XXX: report this annoyance to theano-dev
    """

    errors_handled = [
        'Was not able to allocate output!',
        'expected a CudaNdarray, not None',
    ]

    locals().update(local_vars)

    input = eval(input_exp)
    n_elements = len(input)

    msize = msize_start
    msize_best = msize
    grow_msize = True

    output = None
    i = 0
    while i < n_elements:
        b = time.time()
        if verbose:
            print msize, msize_best

        try:
            exp = '%s = input[i:i+msize]' % slice_exp
            exec(exp)
            slice_output = eval(func_exp)
            msize_best = msize  # it worked with msize
        except Exception, err:
            if verbose:
                print err.message
            # hacky way to detect a out of memory error in theano
            if err.message in errors_handled:
                if verbose:
                    print "!!! Memory error detected: hacking around..."
                exp = '%s = input[i:i+msize_best]' % slice_exp
                eval(exp)
                slice_output = eval(func_exp)
                grow_msize = False
                msize = msize_best
            else:
                raise err

        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        if output is None:
            output = slice_output
        else:
            output = np.concatenate((output, slice_output))

        i += msize

        if verbose:
            print 't: %.3f, msize_best: %d' % (time.time() - b, msize_best)

        if grow_msize:
            msize *= msize_factor

    assert len(output) == len(input), (len(output), len(input))
    assert output.dtype == input.dtype

    return output, msize_best
