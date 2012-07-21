#Minimum pixel error: 0.05486278427454905
#Minimum Rand error: 0.16318234412370003
#Minimum warping error: 0.0020577566964285715

import numpy as np

import theano
from theano import tensor
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d
from scipy import misc
import matplotlib
matplotlib.use("Agg")

from bangmetric.correlation import  pearson
#from bangmetric.precision_recall import average_precision
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
    #factr=1e12,
    maxfun=1e4,
    )


#from sklearn.decomposition import RandomizedPCA


class RPLogReg2(object):

    def __init__(self,
                 rf_size=DEFAULT_RF_SIZE,
                 lnorm_size=DEFAULT_LNORM_SIZE,
                 n_filters=DEFAULT_N_FILTERS,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                 learning=DEFAULT_LEARNING,
                 #pca_n_components=100,
                ):

        self.rf_size = rf_size
        self.lnorm_size = lnorm_size
        self.n_filters = n_filters
        self.learning = learning

        self.fb = None

        self.lbfgs_params = lbfgs_params
        # XXX: seed

        self.fbl = None
        #self.pca = RandomizedPCA(pca_n_components, whiten=True,
                                 #random_state=np.random.RandomState(42))

    def transform(self, X, with_fit=False):

        assert X.ndim == 2

        rf_size = self.rf_size
        lnorm_size = self.lnorm_size
        n_filters = self.n_filters
        X_shape = X.shape
        learning = self.learning

        if lnorm_size is not None:
            X2 = filter_pad2d(np.atleast_3d(X), lnorm_size)
            X2 = np.dstack((
                lcdnorm3(X2, lnorm_size, contrast=False),
                #lcdnorm3(X2, lnorm_size, contrast=True),
            ))

        rf_size = rf_size + (1, )
        X2 = filter_pad2d(X2, rf_size[:2])
        X2 = view_as_windows(X2, rf_size)
        X2 = X2.reshape(np.prod(X.shape[:2]), -1)
        X = X2

        print 'zero-mean / unit-variance'
        zmuv_rows_inplace(X.T)
        #X -= X.mean(0)
        #X /= X.std(0)

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
            print Xnew.shape
            #print 'cast float16'
            #X = X.astype(np.float16)
            #print 'pos'
            #pos = Xnew.clip(0, np.inf) ** 2.
            #Xnew = pos
            #Xnew = X
            #print 'neg'
            #neg = (-Xnew).clip(0, np.inf) ** 2.
            #del X
            #print 'hstack'
            #Xnew = np.hstack((pos, neg))
            #assert np.isfinite(X).all()
            #print X.shape, X.dtype
            #print 'cast float32'
            #X = X.astype(np.float32)

            X = np.column_stack((X, Xnew))

            #print 'pca...'
            #if with_fit:
                #print X.dtype
                #X = self.pca.fit_transform(X).astype('f')
            #else:
                #X = self.pca.transform(X).astype('f')

            print 'zero-mean / unit-variance'
            zmuv_rows_inplace(X.T)
            #Xnew -= Xnew.mean(0)
            #Xnew /= Xnew.std(0)
            assert np.isfinite(X).all()


        X = X.reshape(X_shape[:2] + (-1,))
        print X.shape

        return X

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 2

        #assert Y.dtype == bool

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)
        self.Y = Y

        X = self.transform(X, with_fit=True)
        X = X.reshape(-1, X.shape[-1]).astype('float32')

        #Yv = Y.ravel()
        #pos_mask = Yv > 0
        #pos_idx = np.arange(len(Yv))[pos_mask]
        #neg_idx = np.arange(len(Yv))[~pos_mask]

        print X.shape

        # -- initial variables
        #W = np.ones((X.shape[1], 2), dtype='float32')
        #W_size = W.size
        #W_shape = W.shape
        #b = np.zeros((2), dtype='float32')

        ## -- theano program
        #_X = T.fmatrix()
        #_b = T.fvector()  # could be Theano shared variable
        #_W = T.fmatrix()  # same
        #_Y_true = T.lvector()
        #_pos_idx = T.lvector()
        #_neg_idx = T.lvector()

        #_Y = T.dot(_X, _W) + _b
        #_Y_pred = T.nnet.softmax(_Y)

        #_loss = -T.mean(T.log(_Y_pred)[T.arange(_Y_true.shape[0]), _Y_true])
        ##_loss_pos = -T.mean(T.log(_Y_pred[_pos_idx])[T.arange(_Y_true[_pos_idx].shape[0]), _Y_true[_pos_idx]])
        ##_loss_neg = -T.mean(T.log(_Y_pred[_neg_idx])[T.arange(_Y_true[_neg_idx].shape[0]), _Y_true[_neg_idx]])
        ##_loss = 1. * _loss_pos + 1e-1 * _loss_neg

        #_dloss_W = T.grad(_loss, _W)
        #_dloss_b = T.grad(_loss, _b)

        #_f = theano.function([_X, _W, _b],
                             #[_Y_pred],
                             #allow_input_downcast=True)

        #_f_df = theano.function([_X, _Y_true, _W, _b, _pos_idx, _neg_idx],
                                #[_Y_pred, _loss, _dloss_W, _dloss_b],
                                #allow_input_downcast=True)

        w = np.zeros(X.shape[1], dtype='float32')
        #w = X[10]
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
        t_loss += tensor.mean((1 - t_yb) * tensor.maximum(0, 1 - t_M - m_neg) ** 2.)
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


    def predict(self, X):
        X_shape = X.shape

        X = self.transform(X)
        X = X.reshape(-1, X.shape[-1])

        Y_pred = self._f(X, self.w, self.b)
        Y_pred = Y_pred.reshape(X_shape[:2] + (-1,))

        return Y_pred


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
    tst_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-28.png'
    print tst_fname
    tst_X = (misc.imread(tst_fname, flatten=True) / 255.).astype('f')
    #tst_X -= tst_X.min()
    #tst_X /= tst_X.max()
    tst_X -= tst_X.mean()
    tst_X /= tst_X.std()
    tst_Y = (misc.imread(tst_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')

    # --
    mdl1 = RPLogReg2(rf_size=rf_size,
                     lnorm_size=lnorm_size,
                     n_filters=n_filters,
                     lbfgs_params=lbfgs_params,
                     learning=learning,
                     #pca_n_components=pca_n_components,
                    )
    start = time.time()
    #trn_X -= trn_X.min()
    #trn_Y = trn_Y - (trn_X * trn_X <= 0.1)
    print 'model...'
    from skimage.filter import median_filter
    #trn_X = median_filter(trn_X)
    mdl1.fit(trn_X, trn_Y)
    #trn_X1 = mdl1.predict(trn_X)[..., 0]

    if 0:
        fs = 11
        from bangreadout import LBFGSLogisticClassifier
        val_Y_pred = mdl1.predict(val_X)
        val_Y_pred = np.dstack((val_X[..., np.newaxis], val_Y_pred))
        val_Y_pred_rv = view_as_windows(filter_pad2d(val_Y_pred, (fs, fs)), (fs, fs, 1))
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        val_Y_pred = val_Y_pred_rv.reshape(np.prod(val_Y_pred.shape[:2]), -1)
        n_ff = val_Y_pred.shape[-1]

        logreg = LBFGSLogisticClassifier(n_features=val_Y_pred.shape[-1])
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

        zmuv_rows_inplace(val_Y_pred.T)
        logreg.fit(val_Y_pred, val_Y.ravel()>0)

        Y_pred1 = mdl1.predict(tst_X)
        #zmuv_rows_inplace(Y_pred1.T)
        Y_pred1 = np.dstack((tst_X[..., np.newaxis], Y_pred1))
        Y_pred2 = view_as_windows(filter_pad2d(Y_pred1, (fs, fs)), (fs, fs, 1)).reshape(-1, n_ff)
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        zmuv_rows_inplace(Y_pred2.T)
        Y_pred2 = logreg.transform(Y_pred2).reshape(512, 512)
        Y_pred = Y_pred2
    else:
        Y_pred = mdl1.predict(tst_X)[..., 0]
        Y_pred = median_filter(Y_pred, radius=3)

    #trn_X1 = mdl1.predict(trn_X)[..., 0]

    #mdl2 = RPLogReg1(rf_size=rf_size,
                     #lnorm_size=lnorm_size,
                     #n_filters=n_filters)
    #mdl2 = RPLogReg1(rf_size=rf_size,
                     #lnorm_size=lnorm_size,
                     #n_filters=n_filters)
    #mdl2.fit(trn_X1, trn_Y.astype(bool))

    #Y_pred = mdl2.predict(mdl1.predict(tst_X)[..., 0])
    #print 'pca...'
    #tst_X = pca.transform(tst_X)
    #Y_pred = mdl1.predict(tst_X)[..., 0]
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
    #Y_pred = Y_pred + (tst_X * tst_X <= 0.1)
    Y_true = tst_Y.copy()
    print 'pe =', pearson(Y_true.ravel(), Y_pred.ravel())
    end = time.time()

    print end-start

    from skimage import io
    io.use_plugin('freeimage')
    Y_pred -= Y_pred.min()
    Y_pred /= Y_pred.max()
    offset = 32
    Y_pred = Y_pred[offset:-offset, offset:-offset]
    Y_true = Y_true[offset:-offset, offset:-offset]
    print Y_pred.shape
    io.imsave('Y_pred.tif', Y_pred, plugin='freeimage')
    io.imsave('Y_true.tif', Y_true, plugin='freeimage')

    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

if __name__ == '__main__':
    main()
