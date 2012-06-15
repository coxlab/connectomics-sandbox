import numpy as np

import theano
from theano import tensor as T
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d
from scipy import misc

from bangmetric.correlation import  pearson
from bangmetric.precision_recall import average_precision
from skimage.util.shape import view_as_windows

from scipy.optimize import fmin_l_bfgs_b

DEFAULT_RF_SIZE = (21, 21)
DEFAULT_LNORM_SIZE = (11, 11)
DEFAULT_N_FILTERS = np.prod(DEFAULT_RF_SIZE)

DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1e4,
    )


class RPLogReg1(object):

    def __init__(self,
                 rf_size=DEFAULT_RF_SIZE,
                 lnorm_size=DEFAULT_LNORM_SIZE,
                 n_filters=DEFAULT_N_FILTERS,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.rf_size = rf_size
        self.lnorm_size = lnorm_size
        self.n_filters = n_filters

        self.fb = None

        self.lbfgs_params = lbfgs_params
        # XXX: seed

    def transform(self, X):

        assert X.ndim == 2

        rf_size = self.rf_size
        lnorm_size = self.lnorm_size
        l1_nfilters = self.n_filters
        X_shape = X.shape

        if lnorm_size is not None:
            X = filter_pad2d(np.atleast_3d(X), lnorm_size)
            X = lcdnorm3(X, lnorm_size)[..., 0]

        X = filter_pad2d(np.atleast_3d(X), rf_size)[..., 0]
        X = view_as_windows(X, rf_size)
        X = X.reshape(np.prod(X.shape[:2]), -1)

        print 'zero-mean / unit-variance'
        X -= X.mean(0)
        X /= X.std(0)

        print 'dot...'
        if self.fb is None:
            fb = self.fb = np.random.randn(X.shape[1], l1_nfilters).astype('f')
        else:
            fb = self.fb
        X = np.dot(X, fb).clip(0, np.inf)

        print 'zero-mean / unit-variance'
        X -= X.mean(0)
        X /= X.std(0)

        X = X.reshape(X_shape[:2] + (-1,))

        return X

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 2

        assert Y.dtype is bool

        X = self.transform(X)
        X = X.reshape(-1, X.shape[-1])

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)

        # -- initial variables
        W = np.ones((X.shape[1], 2), dtype='float32')
        W_size = W.size
        W_shape = W.shape
        b = np.zeros((2), dtype='float32')

        # -- theano program
        _X = T.fmatrix()
        _b = T.fvector()
        _W = T.fmatrix()
        _Y_true = T.lvector()

        _Y_pred = T.nnet.softmax(T.dot(_X, _W) + _b)
        _loss = -T.mean(T.log(_Y_pred)[T.arange(_Y_true.shape[0]), _Y_true])

        _dloss_W = T.grad(_loss, _W)
        _dloss_b = T.grad(_loss, _b)

        _f = theano.function([_X, _W, _b],
                             [_Y_pred],
                             allow_input_downcast=True)

        _f_df = theano.function([_X, _Y_true, _W, _b],
                                [_Y_pred, _loss, _dloss_W, _dloss_b],
                                allow_input_downcast=True)

        def func(vars):
            # unpack W and b
            W = vars[:W_size].reshape(W_shape)
            b = vars[W_size:]
            # get loss and gradients from theano function
            Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, W, b)
            # print perf
            try:
                print 'ap', average_precision(Y_true.ravel(), Y_pred[:, 1].ravel())
                print 'pr', pearson(Y_true.ravel(), Y_pred[:, 1].ravel())
            except (AssertionError, ValueError):
                pass
            # pack dW and db
            dloss = np.concatenate([dloss_W.ravel(), dloss_b.ravel()])
            return loss.astype('float64'), dloss.astype('float64')

        vars = np.concatenate([W.ravel(), b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(func, vars, **self.lbfgs_params)

        self.W = best[:W_size].reshape(W_shape)
        self.b = best[W_size:]
        self._f = _f

        return self


    def predict(self, X):
        X_shape = X.shape

        X = self.transform(X)
        X = X.reshape(-1, X.shape[-1])

        Y_pred = self._f(X, self.W, self.b)[0][:, 1]
        Y_pred = Y_pred.reshape(X_shape[:2] + (-1,))

        return Y_pred


def main():

    from coxlabdata.connectomics_hp import ConnectomicsHP
    base_path = '/share/datasets/connectomics/connectomics_hp'

    ds = ConnectomicsHP(base_path)
    m = ds.meta()
    imgdl_ann = [e for e in m if 'annotation' in e]

    rf_size = (21, 21)
    lnorm_size = None#@(11, 11)
    n_filters = np.prod(rf_size) + 100
    #sigma = 1
    #cutoff = .2
    DEBUG = False

    v1 = imgdl_ann[0]
    v1_Y = ds.get_annotation(v1['annotation'][0]).astype('f')
    #v1_Y = (ndi.gaussian_filter(v1_Y, sigma) > cutoff).astype('f')
    #v1_Y = ndi.gaussian_filter(v1_Y, sigma).astype('f')
    #v1_Y -= v1_Y.min()
    #v1_Y /= v1_Y.max()
    v1_X = (misc.imread(v1['filename']) / 255.).astype('f')

    v2 = imgdl_ann[-1]
    v2_Y = ds.get_annotation(v2['annotation'][0]).astype('f')
    #v2_Y = (ndi.gaussian_filter(v2_Y, sigma) > cutoff).astype('f')
    #v2_Y = ndi.gaussian_filter(v2_Y, sigma).astype('f')
    #v2_Y -= v2_Y.min()
    #v2_Y /= v2_Y.max()
    v2_X = (misc.imread(v2['filename']) / 255.).astype('f')

    if DEBUG:
        v1_X = v1_X[::4, ::4]
        v1_Y = v1_Y[::4, ::4]
        v2_X = v2_X[::4, ::4]
        v2_Y = v2_Y[::4, ::4]
    #else:
        #v1_X = v1_X.copy()
        #v1_Y = v1_Y.copy()

    #
    mdl = RPLogReg1(rf_size=rf_size, lnorm_size=lnorm_size, n_filters=n_filters)
    mdl.fit(v1_X, v1_Y)

    Y_pred = mdl.predict(v2_X)
    Y_true = v2_Y.copy()
    print 'pr', pearson(Y_true.ravel(), Y_pred.ravel())

    import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


if __name__ == '__main__':
    main()
