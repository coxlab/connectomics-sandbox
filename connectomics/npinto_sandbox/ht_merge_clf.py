import cPickle as pkl
from glob import glob

import numpy as np
from scipy import ndimage
from scipy import misc
from pprint import pprint

import theano
from theano import tensor
from scipy.optimize import fmin_l_bfgs_b

from skimage import io
io.use_plugin('freeimage')
from skimage.filter import median_filter

from sthor.util.pad import filter_pad2d
from skimage.util.shape import view_as_windows
from bangreadout import zmuv_rows_inplace
from bangreadout import LBFGSLogisticClassifier


def min_max(x):
    o = x.copy()
    o -= o.min()
    o /= o.max()
    return o


from bangmetric.correlation import  pearson
DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    #factr=1e7,
    factr=1e12,
    maxfun=1e4,
    )
class SqHinge(object):

    def __init__(self,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.lbfgs_params = lbfgs_params

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 1

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)
        self.Y = Y

        w = np.zeros(X.shape[1], dtype='float32')
        w_size = w.size
        b = np.zeros(1, dtype='float32')

        Y_true = 2. * Y_true - 1

        m = 0.2

        def theano_pearson_normalize_vector(X):
            Xm = X - X.mean()
            Xmn = Xm / (tensor.sqrt(tensor.dot(Xm, Xm)) + 1e-3)
            return Xmn

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
        #t_loss += tensor.mean((1 - t_yb) * tensor.maximum(0, 1 - t_M - m_neg) ** 2.)
        t_loss += tensor.mean((1 - t_yb) * tensor.maximum(0, 1 - t_M - m_neg))
        #t_loss = 1 - tensor.dot(
            #theano_pearson_normalize_vector(t_y.flatten()),
            #theano_pearson_normalize_vector(t_H.flatten())
            #)
        #from bangreadout.util import theano_corrcoef
        #t_loss = 1 - theano_corrcoef(t_y.flatten(), t_H.flatten())
        #t_loss = 1 - theano_pearson_normalize(t_y.ravel(), t_H.ravel())

        #t_loss = tensor.mean(tensor.maximum(0, 1 - t_M - m) ** 2.)
        ##t_loss = tensor.mean((1 - t_M) ** 2.)
        #t_loss = tensor.mean(tensor.maximum(0, 1 - t_M - m))
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




of = 32
rf_size = (21, 21, 1)
rf_stride = 2
n_models = 10
t_l = 'px', 'rd'#, 'wp'


fl = []
for t in t_l:
    for i in xrange(n_models):
        fl += ['top_models/model_%s_%d.pkl' % (t, i)]

#
print 'pkl_l with %d models...' % len(fl)
pprint(fl)
pkl_l = [pkl.load(open(fn)) for fn in fl]

print 'X_{val,tst}_l'
X_val_l = [p['val'][0] for p in pkl_l]
X_tst_l = [p['tst'][0] for p in pkl_l]
X_val0 = (misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-03.png', flatten=True)).astype('f')

print 'Y_{trn,tst}'
Y_trn = (misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-labels.tif-03.png', flatten=True) > 0).astype('f')
Y_tst = (misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-labels.tif-29.png', flatten=True) > 0).astype('f')
X_tst0 = (misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png', flatten=True)).astype('f')

print '*' * 80
print 'TRAINING'
print '*' * 80

print 'dstack'
X_trn = np.dstack([np.atleast_3d(e) for e in X_val_l] + [X_val0])
X_trn = filter_pad2d(X_trn, rf_size[:2])
X_trn = view_as_windows(X_trn, rf_size)
X_trn = X_trn[..., ::rf_stride, ::rf_stride, 0]

print 'reshape'
X_trn = X_trn.reshape(np.prod(X_trn.shape[:2]), -1)
print X_trn.shape
zmuv_rows_inplace(X_trn.T)

print 'clf.fit'
#clf = LBFGSLogisticClassifier(X_trn.shape[1])
clf = SqHinge()#X_trn.shape[1])
clf.fit(X_trn, Y_trn.ravel()>0)

print '*' * 80
print 'TESTING'
print '*' * 80

print 'dstack'
X_tst = np.dstack([np.atleast_3d(e) for e in X_tst_l] + [X_tst0])
X_tst = filter_pad2d(X_tst, rf_size[:2])
X_tst = view_as_windows(X_tst, rf_size)
X_tst = X_tst[..., ::rf_stride, ::rf_stride, 0]

print 'reshape'
X_tst = X_tst.reshape(np.prod(X_tst.shape[:2]), -1)
print X_tst.shape
zmuv_rows_inplace(X_tst.T)

print 'clf.transform'
Y_pred = clf.transform(X_tst)
Y_pred = Y_pred.reshape(512, 512)
Y_pred = median_filter(Y_pred, radius=2)

Y_pred = min_max(Y_pred[of:-of, of:-of])
Y_true = min_max(Y_tst[of:-of, of:-of])

print 'pe:', pearson(Y_true.ravel(), Y_pred.ravel())

io.imsave('ht_merge_clf_Y_true.tif', Y_true)
io.imsave('ht_merge_clf_Y_pred.tif', Y_pred)

#r2 = np.array([min_max(e)[of:-of, of:-of] for e in r])
#io.imsave('median9.tif', min_max(np.median(r2, axis=0)))
