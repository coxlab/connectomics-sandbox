from copy import deepcopy
import numpy as np
#from scipy import signal
from theano.tensor import nnet
from theano.tensor.signal import downsample
from scipy import misc
import theano
from theano import tensor
from sthor.util import arraypad
from skimage.util import view_as_windows

l = (misc.lena() / 1.).astype('f')


desc = [
    (16, 5, 2),
    (16, 5, 2),
    #(16, 5, 2),
]


class SharpMind(object):

    def __init__(self, description):

        # -- model description
        for i, (nf, fsize, psize) in enumerate(desc):
            print(">>> Layer %d: fsize=%d, psize=%d"
                  % ((i + 1), fsize, psize))
            assert fsize % 1 == 0, fsize
            assert psize % 2 == 0, psize
        self.description = description

        # -- footprint / padding
        footprint = 1
        for _, fsize, psize in desc[::-1]:
            footprint *= psize
            footprint += fsize - 1
        assert footprint > 0, footprint
        self.footprint = footprint
        print ">>> Footprint: %d" % footprint

    def partial_fit(self, X, Y):

        assert X.ndim == 2
        assert X.dtype == 'float32'

        assert Y.ndim == 2
        assert Y.dtype == bool

        # -- reshape X to fit theano's convention
        print '>>> X.shape:', X.shape
        X = X.reshape((1, 1) + X.shape)
        print '>>> X.shape:', X.shape

        # -- theano stuff
        t_X = tensor.ftensor4()

        fb_l = []

        t_input_l = []
        input_l = []
        t_fb_l = []
        t_f_l = []
        t_p_l = []
        t_output_l = []

        output_shape = X.shape
        t_output = t_X
        for i, (nf, fsize, psize) in enumerate(desc):
            input_shape = output_shape
            fb = np.random.randn(nf, input_shape[1], fsize, fsize).astype('f')

            t_input = t_output
            t_fb = tensor.ftensor4()
            t_f = tensor.tanh(nnet.conv2d(t_input, t_fb))
            t_p = downsample.max_pool_2d(t_f, (psize, psize))
            t_output = t_p

            output_shape = (
                input_shape[0],
                nf,
                (input_shape[2] - fsize + 1) / psize,
                (input_shape[3] - fsize + 1) / psize
            )

            fsize2 = fsize // 2
            Y = Y[fsize2:-fsize2, fsize2:-fsize2]
            Y = Y[::psize, ::psize]
            assert Y.shape == output_shape[2:], (Y.shape, output_shape)

            input_l += [input]
            t_input_l += [t_input]
            fb_l += [fb]
            t_fb_l += [t_fb]
            t_f_l += [t_f]
            t_p_l += [t_p]
            t_output_l += [t_output]

        print '>>> Compiling theano function...'
        f = theano.function([t_X] + t_fb_l, t_output)
        self.f = f

m = SharpMind(desc)
pad = m.footprint

l = np.random.randn(pad, pad).astype('f')

m.partial_fit(l, l>100)

#print X.shape
#print [fb.shape for fb in fb_l]
#params = (X,) + tuple(fb_l)
#Y = f(*params)
#print Y.shape
#print
