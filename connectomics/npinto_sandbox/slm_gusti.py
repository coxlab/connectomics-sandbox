import numpy as np

import theano
from theano import tensor
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d
from scipy import misc
import matplotlib
matplotlib.use("Agg")

from bangmetric.correlation import  pearson
from skimage.util.shape import view_as_windows
from bangreadout import zmuv_rows_inplace

import time

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


class LogRegG(object):

    def __init__(self,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.lbfgs_params = lbfgs_params

    def partial_fit(self, X, Y):
        return self.fit(X, Y)

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 1

        assert len(X) == len(Y)

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)
        self.Y = Y

        w = np.zeros(X.shape[1], dtype='float32')
        w_size = w.size
        b = np.zeros(1, dtype='float32')

        Y_true = 2. * Y_true - 1

        m = 0.2

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
            #Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, w, b)
            Y_pred, loss, dloss_w, dloss_b = _f_df(X, Y_true, w, b[0])
            try:
                #print 'ap', average_precision(Y_true.ravel(), Y_pred[:, 1].ravel())
                print 'pe =', pearson(Y_true.ravel(), Y_pred.ravel())
            except (AssertionError, ValueError):
                pass
            # pack dW and db
            dloss = np.concatenate([dloss_w.ravel(), dloss_b.ravel()])
            return loss.astype('float64'), dloss.astype('float64')

        vars = np.concatenate([w.ravel(), b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(func, vars, **self.lbfgs_params)

        #self.W = best[:W_size].reshape(W_shape)
        #self.b = best[W_size:]
        self.w = best[:w_size]
        self.b = best[w_size:][0]
        self._f = _f

        return self


    def transform(self, X):
        #X_shape = X.shape

        Y_pred = self._f(X, self.w, self.b)
        #Y_pred = Y_pred.reshape(X_shape[:2] + (-1,))

        return Y_pred

    def predict(self, X):
        return self.transform(X)


def main():

    #rf_size = (51, 51)
    rf_size = (21, 21)
    lnorm_size = (11, 11)
    n_filters = 512
    #learning = 'randn'
    learning = 'imprint'
    #DEBUG = False
    N_IMGS = 1
    N_IMGS_VAL = 1
    #pca_n_components = 100

    np.random.seed(42)

    lbfgs_params = dict(
        iprint=1,
        factr=1e12,
        maxfun=1000,
        #maxfun=2,
        #factr=1e7,
        #maxfun=1e4,
        )

    print 'training image'
    trn_X_l = []
    trn_Y_l = []
    for i in range(N_IMGS):
        trn_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % i
        print trn_fname
        trn_X = (misc.imread(trn_fname, flatten=True) / 255.).astype('f')
        #trn_X -= trn_X.min()
        #trn_X /= trn_X.max()
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
        val_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % k
        print val_fname
        val_X = (misc.imread(val_fname, flatten=True) / 255.).astype('f')
        #val_X -= val_X.min()
        #val_X /= val_X.max()
        val_X -= val_X.mean()
        val_X /= val_X.std()
        val_Y = (misc.imread(val_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')
        val_X_l += [val_X]
        val_Y_l += [val_Y]
    val_X = np.array(val_X_l).reshape(N_IMGS_VAL*512, 512)
    val_Y = np.array(val_Y_l).reshape(N_IMGS_VAL*512, 512)


    print 'testing image'
    tst_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png'
    print tst_fname
    tst_X = (misc.imread(tst_fname, flatten=True) / 255.).astype('f')
    #tst_X -= tst_X.min()
    #tst_X /= tst_X.max()
    tst_X -= tst_X.mean()
    tst_X /= tst_X.std()
    tst_Y = (misc.imread(tst_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')

    # --
    #mdl1 = RPLogReg2(rf_size=rf_size,
                     #lnorm_size=lnorm_size,
                     #n_filters=n_filters,
                     #lbfgs_params=lbfgs_params,
                     #learning=learning,
                     ##pca_n_components=pca_n_components,
                    #)
    #start = time.time()
    #trn_X -= trn_X.min()
    #trn_Y = trn_Y - (trn_X * trn_X <= 0.1)
    #from sthor.model import parameters
    from sthor.model import slm
    #from bangreadout import logistic
    from sthor import util
    import parameters

    # XXXX: HERE XXXX
    desc = parameters.gusti1
    m = slm.SequentialLayeredModel(trn_X.shape, desc)

    #if lnorm_size is not None:
        #trn_X = filter_pad2d(np.atleast_3d(trn_X), lnorm_size)
        #trn_X = np.dstack((
            #lcdnorm3(trn_X, lnorm_size, contrast=False),
        #))[..., 0]

        #tst_X = filter_pad2d(np.atleast_3d(tst_X), lnorm_size)
        #tst_X = np.dstack((
            #lcdnorm3(tst_X, lnorm_size, contrast=False),
        #))[..., 0]

    print 'pad'
    X = trn_X
    X = util.arraypad.pad(trn_X, (np.array(m.receptive_field_shape) // 2)[0], mode='symmetric')
    m = slm.SequentialLayeredModel(X.shape, desc)
    r = m.transform(X, interleave_stride=True, pad_apron=True)
    r /= r.sum(2)[..., np.newaxis]
    offset = (r.shape[0] - 512) / 2
    r = r[offset:-offset, offset:-offset]

    #c = logistic.AverageLBFGSLogisticClassifier(r.shape[-1])
    c = LogRegG()
    a = r.reshape(-1, r.shape[-1])
    b = (trn_Y>0).ravel()
    #zmuv_rows_inplace(a.T)
    c.partial_fit(a, b)
    print 'trn pe', pearson(b.ravel(), c.transform(a).ravel())

    X_tst_pad = util.arraypad.pad(
        tst_X, (np.array(m.receptive_field_shape) // 2)[0], mode='symmetric')
    X_tst_r = m.transform(X_tst_pad, interleave_stride=True, pad_apron=True)
    X_tst_r /= X_tst_r.sum(2)[..., np.newaxis]
    X_tst_r = X_tst_r[offset:-offset, offset:-offset]
    a = X_tst_r.reshape(-1, r.shape[-1])
    b = (tst_Y>0).ravel()
    print 'tst pe', pearson(b.ravel(), c.transform(a).ravel())

    import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

if __name__ == '__main__':
    main()

