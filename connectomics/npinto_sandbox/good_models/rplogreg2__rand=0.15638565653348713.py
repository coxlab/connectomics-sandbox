#Minimum pixel error: 0.05486278427454905
#Minimum Rand error: 0.16318234412370003
#Minimum warping error: 0.0020577566964285715

import numpy as np

from os import path, environ
home = environ.get("HOME")

import theano
from theano import tensor
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d
from scipy import misc
import matplotlib
matplotlib.use("Agg")

from bangmetric.correlation import  pearson
from bangmetric.isbi12 import rand_error
from skimage.util.shape import view_as_windows
from bangreadout import zmuv_rows_inplace
from skimage.filter import median_filter

import time

from skimage import io
io.use_plugin('freeimage')

from scipy.optimize import fmin_l_bfgs_b

DEFAULT_RF_SIZE = (21, 21)
DEFAULT_LNORM_SIZE = (11, 11)
DEFAULT_N_FILTERS = np.prod(DEFAULT_RF_SIZE)
DEFAULT_LEARNING = 'randn'

DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1e4,
    )


class RPLogReg2(object):

    def __init__(self,
                 rf_size=DEFAULT_RF_SIZE,
                 lnorm_size=DEFAULT_LNORM_SIZE,
                 n_filters=DEFAULT_N_FILTERS,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                 learning=DEFAULT_LEARNING,
                ):

        self.rf_size = rf_size
        self.lnorm_size = lnorm_size
        self.n_filters = n_filters
        self.learning = learning

        self.fb = None

        self.lbfgs_params = lbfgs_params
        # XXX: seed

        self.fbl = None

    def transform(self, X, with_fit=False):

        assert X.ndim == 2

        rf_size = self.rf_size
        lnorm_size = self.lnorm_size
        n_filters = self.n_filters
        X_shape = X.shape
        learning = self.learning

        if lnorm_size is not None:
            # TODO: arraypad
            X2 = filter_pad2d(np.atleast_3d(X), lnorm_size)
            X2 = np.dstack((
                lcdnorm3(X2, lnorm_size, contrast=False),
            ))

        rf_size = rf_size + (1, )
        X2 = filter_pad2d(X2, rf_size[:2])
        X2 = view_as_windows(X2, rf_size)
        X2 = X2.reshape(np.prod(X.shape[:2]), -1)

        X = X2

        print 'zero-mean / unit-variance'
        zmuv_rows_inplace(X.T)

        if n_filters > 0:
            if self.fb is None:
                print "'learning' with %s..." % learning
                if learning == 'randn':
                    fb = self.fb = np.random.randn(X.shape[1], n_filters).astype('f')
                elif learning == 'imprint':
                    ridx = np.random.permutation(len(X))[:n_filters]
                    fb = self.fb = X[ridx].T.copy()
                else:
                    raise ValueError("'%s' learning not understood"
                                     % learning)
            else:
                fb = self.fb
            print 'dot...'
            Xnew = np.dot(X, fb) ** 2.

            X = np.column_stack((X, Xnew))

            print 'zero-mean / unit-variance'
            zmuv_rows_inplace(X.T)

            assert np.isfinite(X).all()

        X = X.reshape(X_shape[:2] + (-1,))
        print X.shape

        return X

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 2

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)
        self.Y = Y

        X = self.transform(X, with_fit=True)
        X = X.reshape(-1, X.shape[-1]).astype('float32')

        print X.shape

        # --
        w = np.zeros(X.shape[1], dtype='float32')
        w_size = w.size
        b = np.zeros(1, dtype='float32')

        Y_true = 2. * Y_true - 1

        # -- theano variables
        t_X = tensor.fmatrix()
        t_y = tensor.fvector()
        t_w = tensor.fvector()
        t_b = tensor.fscalar()

        t_H = tensor.dot(t_X, t_w) + t_b
        t_H = 2. * tensor.nnet.sigmoid(t_H) - 1

        t_M = t_y * t_H

        t_yb = (t_y + 1) / 2
        m_pos = 0.5
        m_neg = 0
        t_loss = tensor.mean((t_yb * (tensor.maximum(0, 1 - t_M - m_pos) ** 2.)))
        t_loss += tensor.mean((1 - t_yb) * tensor.maximum(0, 1 - t_M - m_neg) ** 2.)

        t_dloss_dw = tensor.grad(t_loss, t_w)
        t_dloss_db = tensor.grad(t_loss, t_b)

        print 'compiling theano functions...'
        _f = theano.function(
            [t_X, t_w, t_b],
            t_H,
            allow_input_downcast=True)

        _f_df = theano.function(
            [t_X, t_y, t_w, t_b],
            [t_H, t_loss, t_dloss_dw, t_dloss_db],
            allow_input_downcast=True)

        def func(vars):
            # unpack W and b
            w = vars[:w_size]
            b = vars[w_size:]
            # get loss and gradients from theano function
            Y_pred, loss, dloss_w, dloss_b = _f_df(X, Y_true, w, b[0])
            try:
                print 'pe =', pearson(Y_true.ravel(), Y_pred.ravel())
            except (AssertionError, ValueError):
                pass
            # pack dW and db
            dloss = np.concatenate([dloss_w.ravel(), dloss_b.ravel()])
            return loss.astype('float64'), dloss.astype('float64')

        vars = np.concatenate([w.ravel(), b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(func, vars, **self.lbfgs_params)

        self.w = best[:w_size]
        self.b = best[w_size:][0]
        self._f = _f

        return self


    def predict(self, X):
        X_shape = X.shape

        X = self.transform(X)
        X = X.reshape(-1, X.shape[-1])

        Y_pred = self._f(X, self.w, self.b)
        Y_pred = Y_pred.reshape(X_shape[:2] + (-1,))

        return Y_pred


def main():

    rf_size = (21, 21)
    lnorm_size = (11, 11)
    n_filters = 512
    learning = 'imprint'
    N_IMGS = 1
    N_IMGS_VAL = 1

    np.random.seed(42)

    lbfgs_params = dict(
        iprint=1,
        factr=1e12,
        maxfun=1000,
        )

    print 'training image'
    trn_X_l = []
    trn_Y_l = []
    for i in range(N_IMGS):
        trn_fname = path.join(
            home, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % i)
        print trn_fname
        trn_X = (misc.imread(trn_fname, flatten=True) / 255.).astype('f')
        trn_X -= trn_X.mean()
        trn_X /= trn_X.std()
        trn_Y = (misc.imread(trn_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')
        trn_X_l += [trn_X]
        trn_Y_l += [trn_Y]
    trn_X = np.array(trn_X_l).reshape(N_IMGS*512, 512)
    trn_Y = np.array(trn_Y_l).reshape(N_IMGS*512, 512)

    print 'validation image'
    val_X_l = []
    val_Y_l = []
    for j in range(N_IMGS_VAL):
        k = j + i + 1
        val_fname = path.join(
            home, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % k)
        print val_fname
        val_X = (misc.imread(val_fname, flatten=True) / 255.).astype('f')
        val_X -= val_X.mean()
        val_X /= val_X.std()
        val_Y = (misc.imread(val_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')
        val_X_l += [val_X]
        val_Y_l += [val_Y]
    val_X = np.array(val_X_l).reshape(N_IMGS_VAL*512, 512)
    val_Y = np.array(val_Y_l).reshape(N_IMGS_VAL*512, 512)


    print 'testing image'
    tst_fname = path.join(home, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png')
    print tst_fname
    tst_X = (misc.imread(tst_fname, flatten=True) / 255.).astype('f')
    tst_X -= tst_X.mean()
    tst_X /= tst_X.std()
    tst_Y = (misc.imread(tst_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')

    # --
    mdl1 = RPLogReg2(rf_size=rf_size,
                     lnorm_size=lnorm_size,
                     n_filters=n_filters,
                     lbfgs_params=lbfgs_params,
                     learning=learning,
                    )
    start = time.time()
    print 'model...'
    mdl1.fit(trn_X, trn_Y)
    Y_pred = mdl1.predict(tst_X)[..., 0]
    Y_pred = median_filter(Y_pred, radius=3)

    Y_true = tst_Y.copy()
    print 'pe =', pearson(Y_true.ravel(), Y_pred.ravel())
    end = time.time()

    print end-start

    Y_pred -= Y_pred.min()
    Y_pred /= Y_pred.max()
    offset = 32
    Y_pred = Y_pred[offset:-offset, offset:-offset]
    Y_true = Y_true[offset:-offset, offset:-offset]
    print 'rd =', rand_error(Y_true, Y_pred)
    print Y_pred.shape
    io.imsave('Y_pred.tif', Y_pred, plugin='freeimage')
    io.imsave('Y_true.tif', Y_true, plugin='freeimage')


if __name__ == '__main__':
    main()
