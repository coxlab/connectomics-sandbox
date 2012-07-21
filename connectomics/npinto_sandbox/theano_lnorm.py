import numpy as np

import theano
from theano import tensor
from theano.tensor import nnet

from sthor.operation.lnorm import (
    EPSILON,
    DEFAULT_STRIDE,
    DEFAULT_THRESHOLD,
    DEFAULT_STRETCH,
    DEFAULT_CONTRAST,
    DEFAULT_DIVISIVE,
    )


def theano_lsum(t_arr_in, neighborhood):
    ker = np.ones(neighborhood, dtype='float32')

    nnet.conv2d()
                #conv.conv2d(x,
                    #kerns,
                    #image_shape=x_shp,
                    #filter_shape=kerns.shape,
                    #border_mode='valid'),
                #rshp)
    #except Exception, e:
        #if "Bad size for the output shape" in str(e):

def theano_lcdnorm3(t_arr_in, neighborhood,
                    contrast=DEFAULT_CONTRAST,
                    divisive=DEFAULT_DIVISIVE,
                    stretch=DEFAULT_STRETCH,
                    threshold=DEFAULT_THRESHOLD,
                    stride=DEFAULT_STRIDE
                   ):

    assert t_arr_in.ndim == 3
    assert len(neighborhood) == 2
    assert isinstance(contrast, bool)
    assert isinstance(divisive, bool)
    assert contrast or divisive

    inh, inw, ind = t_arr_in.shape

    nbh, nbw = neighborhood
    assert nbh <= inh
    assert nbw <= inw

    nb_size = 1. * nbh * nbw * ind

    # -- prepare arr_out
    lys = nbh / 2
    lxs = nbw / 2
    rys = (nbh - 1) / 2
    rxs = (nbw - 1) / 2
    t_arr_out = t_arr_in[lys:inh-rys, lxs:inw-rxs][::stride, ::stride]

    # -- Contrast Normalization
    if contrast:

        # -- local sums
        t_arr_sum = theano_lsum(t_arr_in, neighborhood)

        # -- remove the mean
        t_arr_out = t_arr_out - t_arr_sum / nb_size

    # -- Divisive (gain) Normalization
    if divisive:

        # -- local sums of squares
        t_arr_ssq = theano_lsum(t_arr_in ** 2., neighborhood)

        # -- divide by the euclidean norm
        if contrast:
            t_l2norms = (t_arr_ssq - (t_arr_sum ** 2.0) / nb_size)
        else:
            t_l2norms = t_arr_ssq

        t_l2norms = tensor.maximum(0, t_l2norms)
        t_l2norms = tensor.sqrt(t_l2norms) + EPSILON

        if stretch != 1:
            t_arr_out *= stretch
            t_l2norms *= stretch

        t_l2norms = tensor.switch(t_l2norms < (threshold + EPSILON), 1.0, t_l2norms)
        t_arr_out = t_arr_out / t_l2norms

    return t_arr_out

def main():
    pass

if __name__ == '__main__':
    main()

#def boxconv(self, x, x_shp, kershp, channels=False):
    #"""
    #channels: sum over channels (T/F)
    #"""
    #kershp = tuple(kershp)
    #if channels:
        #rshp = (   x_shp[0],
                    #1,
                    #x_shp[2] - kershp[0] + 1,
                    #x_shp[3] - kershp[1] + 1)
        #kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
    #else:
        #rshp = (   x_shp[0],
                    #x_shp[1],
                    #x_shp[2] - kershp[0] + 1,
                    #x_shp[3] - kershp[1] + 1)
        #kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
        #x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
        #x = x.reshape(x_shp)
    #try:
        #rval = tensor.reshape(
                #conv.conv2d(x,
                    #kerns,
                    #image_shape=x_shp,
                    #filter_shape=kerns.shape,
                    #border_mode='valid'),
                #rshp)
    #except Exception, e:
        #if "Bad size for the output shape" in str(e):
            #raise InvalidDescription()
        #else:
            #raise
    #return rval, rshp

#def init_lnorm_h(self, x, x_shp, **kwargs):
    #threshold = kwargs.get('threshold', DEFAULT_THRESHOLD)
    #stretch = kwargs.get('stretch', DEFAULT_STRETCH)
    #kwargs['threshold'] = get_into_shape(threshold)
    #kwargs['stretch'] = get_into_shape(stretch)
    #return self.init_lnorm(x, x_shp, **kwargs)


#def init_lnorm(self, x, x_shp,
        #inker_shape=DEFAULT_INKER_SHAPE,    # (3, 3)
        #outker_shape=DEFAULT_OUTKER_SHAPE,  # (3, 3)
        #remove_mean=DEFAULT_REMOVE_MEAN,    # False
        #div_method=DEFAULT_DIV_METHOD,      # 'euclidean'
        #threshold=DEFAULT_THRESHOLD,        # 0.
        #stretch=DEFAULT_STRETCH,            # 1.
        #mode=DEFAULT_MODE,                  # 'valid'
        #):
    #if mode != 'valid':
        #raise NotImplementedError('lnorm requires mode=valid', mode)

    #if outker_shape == inker_shape:
        #size = np.asarray(x_shp[1] * inker_shape[0] * inker_shape[1],
                #dtype=x.dtype)
        #ssq, ssqshp = self.boxconv(x ** 2, x_shp, inker_shape,
                #channels=True)
        #xs = inker_shape[0] // 2
        #ys = inker_shape[1] // 2
        ## --local contrast normalization in regions that are not symmetric
        ##   about the pixel being normalized feels weird, but we're
        ##   allowing it here.
        #xs_inc = (inker_shape[0] + 1) % 2
        #ys_inc = (inker_shape[1] + 1) % 2
        #if div_method == 'euclidean':
            #if remove_mean:
                #arr_sum, _shp = self.boxconv(x, x_shp, inker_shape,
                        #channels=True)
                #arr_num = (x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                        #- arr_sum / size)
                #arr_div = EPSILON + tensor.sqrt(
                        #tensor.maximum(0,
                            #ssq - (arr_sum ** 2) / size))
            #else:
                #arr_num = x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                #arr_div = EPSILON + tensor.sqrt(ssq)
        #else:
            #raise NotImplementedError('div_method', div_method)
    #else:
        #raise NotImplementedError('outker_shape != inker_shape',outker_shape, inker_shape)
    #if (hasattr(stretch, '__iter__') and (stretch != 1).any()) or stretch != 1:
        #arr_num = arr_num * stretch
        #arr_div = arr_div * stretch
    #arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

    #r = arr_num / arr_div
    #r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
    #return r, r_shp
