import numpy as np

import theano
from theano import tensor as T
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d
from scipy import misc

from bangmetric.correlation import  pearson
#from bangmetric.precision_recall import average_precision
from skimage.util.shape import view_as_windows

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


class RPLogReg1(object):

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

    def transform(self, X):

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
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        X = X2

        print 'zero-mean / unit-variance'
        X -= X.mean(0)
        X /= X.std(0)

        # --
        if False:

            m = n_filters / 2
            q = 256

            X_new = np.empty((X.shape[0], m*2), dtype='float32')

            #m = n_filters
            if self.fbl is None:
                self.fbl = np.random.randn(m, X.shape[1], q).astype('f')
            print 'fbl mean', self.fbl.mean()

            for fi, fb in enumerate(self.fbl):
                if fi % 10 == 0:
                    print 'q-ary', fi
                qdot = np.dot(X, fb)
                amax = qdot.argmax(1)
                amin = qdot.argmin(1)
                X_new[:, 2 * fi] = amax
                X_new[:, 2 * fi + 1] = amin

            X = X_new
            X = X.reshape(X_shape[:2] + (-1,))
            #X = X.astype(np.float32)
            print 'zero-mean / unit-variance'
            X -= X.mean(0)
            X_std = X.std(0)
            np.putmask(X_std, X_std < 1e-3, 0.)
            X /= X_std
            assert np.isfinite(X).all()

            return X


        if n_filters > 0:
            if self.fb is None:
                print "'learning' with %s..." % learning
                if learning == 'randn':
                    fb = self.fb = np.random.randn(X.shape[1], n_filters).astype('f')
                elif learning == 'imprint':
                    ridx = np.random.permutation(len(X))[:n_filters]
                    fb = self.fb = X[ridx].T.copy()
                elif learning == 'imprint_pos':
                    Y = self.Y.ravel()
                    ridx = np.arange(len(X))[Y > 0]
                    print len(ridx), (Y > 0).sum(), Y.size
                    np.random.shuffle(ridx)
                    ridx = ridx[:n_filters]
                    fb = X[ridx].T.copy()
                    done = False
                    loss_min = np.inf
                    iter = 0
                    best_fb = fb.copy()
                    while not done:
                        fb = best_fb.copy()
                        # select _ filter to replace
                        for _ in xrange(1):#np.random.randint(len(fb)-1)+1):
                            fi = np.random.randint(len(ridx))
                            rii = np.random.randint(len(ridx))
                            fb[:, fi] = X[rii].copy() + np.random.randn(*X[rii].shape)
                        #for fi in xrange(len(fb)):
                            #f = fb[:, fi]
                            #f -= f.mean()
                            #f /= np.linalg.norm(f)
                            #fb[:, fi] = f.copy()
                        #print 'mean', fb.mean()
                        #np.random.shuffle(ridx)
                        #ridx = ridx[:n_filters]
                        #print ridx
                        ##fb = X.copy()[ridx].T.copy()
                        #fb = np.take(X.copy(), ridx).T.copy()
                        #print np.corrcoef(fb)
                        #loss = np.linalg.norm((np.corrcoef(fb).clip(0, np.inf) - np.eye(len(fb))), 1)
                        loss_independence = np.mean((np.corrcoef(fb).clip(0, np.inf) - np.eye(len(fb))))
                        print 'loss_independence', loss_independence
                        coef_independence = 100
                        loss_coverage = - np.dot(X[Y > 0][::10], fb).clip(0, np.inf).max(1).mean()
                        print 'loss_coverage', loss_coverage
                        coef_coverage = 0.5
                        loss = coef_independence * loss_independence + coef_coverage * loss_coverage
                        if loss < loss_min:
                            loss_min = loss
                            best_fb = fb.copy()
                        iter += 1

                        print '#%d, curr=%s, best=%s' % (iter, loss, loss_min)

                        #print 'score', score
                        if iter >= 1000:
                            break

                    #fb = self.fb = X[ridx].T.copy()
                    fb = self.fb = best_fb
                else:
                    raise ValueError("'%s' learning not understood"
                                     % learning)
            else:
                fb = self.fb
            print 'dot...'
            Xnew = np.dot(X, fb)
            print Xnew.shape
            #print 'cast float16'
            #X = X.astype(np.float16)
            print 'pos'
            pos = Xnew.clip(0, np.inf)
            Xnew = pos
            #print 'neg'
            #neg = (-X).clip(0, np.inf)
            #del X
            #print 'hstack'
            #X = np.hstack((pos, neg))
            #assert np.isfinite(X).all()
            #print X.shape, X.dtype
            #print 'cast float32'
            #X = X.astype(np.float32)

            print 'zero-mean / unit-variance'
            Xnew -= Xnew.mean(0)
            Xnew /= Xnew.std(0)
            X = np.column_stack((X, Xnew))
            assert np.isfinite(X).all()


        #if n_filters > 0:
            #print "'learning' with %s..." % learning
            #if self.fb is None:
                #if learning == 'randn':
                    #fb = self.fb = np.random.randn(X.shape[1], n_filters).astype('f')
                #elif learning == 'imprint':
                    #ridx = np.random.permutation(len(X))[:n_filters]
                    #fb = self.fb = X[ridx].T.copy()
                    ##fb -= fb.mean(0)
                    ##fb /= fb.std(0)
                    ##fb = self.fb = fb.T
                #else:
                    #raise ValueError("'%s' learning not understood"
                                     #% learning)
            #else:
                #fb = self.fb
            #print 'dot...'
            #X = np.dot(X, fb)
            ##print X.shape
            ###print 'cast float16'
            ###X = X.astype(np.float16)
            #print 'pos'
            #pos = X.clip(0, np.inf)
            ##X = pos
            ###print 'neg'
            ###neg = (-X).clip(0, np.inf)
            ###del X
            ###print 'hstack'
            ###X = np.hstack((pos, neg))
            ###assert np.isfinite(X).all()
            ###print X.shape, X.dtype
            ###print 'cast float32'
            ###X = X.astype(np.float32)

            #print 'zero-mean / unit-variance'
            #X -= X.mean(0)
            #X /= X.std(0)
            #assert np.isfinite(X).all()

        X = X.reshape(X_shape[:2] + (-1,))
        #X = X.astype(np.float32)

        return X

    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 2

        assert Y.dtype == bool

        Y = Y.reshape(Y.size, 1)
        Y_true = Y.ravel().astype(long)
        self.Y = Y

        X = self.transform(X)
        X = X.reshape(-1, X.shape[-1]).astype('float32')

        Yv = Y.ravel()
        pos_mask = Yv > 0
        pos_idx = np.arange(len(Yv))[pos_mask]
        neg_idx = np.arange(len(Yv))[~pos_mask]

        print X.shape

        # -- initial variables
        W = np.ones((X.shape[1], 2), dtype='float32')
        W_size = W.size
        W_shape = W.shape
        b = np.zeros((2), dtype='float32')

        # -- theano program
        _X = T.fmatrix()
        _b = T.fvector()  # could be Theano shared variable
        _W = T.fmatrix()  # same
        _Y_true = T.lvector()
        _pos_idx = T.lvector()
        _neg_idx = T.lvector()

        _Y = T.dot(_X, _W) + _b
        _Y_pred = T.nnet.softmax(_Y)

        #_loss = -T.mean(T.log(_Y_pred)[T.arange(_Y_true.shape[0]), _Y_true])
        _loss_pos = -T.mean(T.log(_Y_pred[_pos_idx])[T.arange(_Y_true[_pos_idx].shape[0]), _Y_true[_pos_idx]])
        _loss_neg = -T.mean(T.log(_Y_pred[_neg_idx])[T.arange(_Y_true[_neg_idx].shape[0]), _Y_true[_neg_idx]])
        _loss = 1. * _loss_pos + 1e-1 * _loss_neg

        _dloss_W = T.grad(_loss, _W)
        _dloss_b = T.grad(_loss, _b)

        _f = theano.function([_X, _W, _b],
                             [_Y_pred],
                             allow_input_downcast=True)

        _f_df = theano.function([_X, _Y_true, _W, _b, _pos_idx, _neg_idx],
                                [_Y_pred, _loss, _dloss_W, _dloss_b],
                                allow_input_downcast=True)

        def func(vars):
            # unpack W and b
            W = vars[:W_size].reshape(W_shape)
            b = vars[W_size:]
            # get loss and gradients from theano function
            Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, W, b, pos_idx, neg_idx)
            try:
                #print 'ap', average_precision(Y_true.ravel(), Y_pred[:, 1].ravel())
                print 'pe =', pearson(Y_true.ravel(), Y_pred[:, 1].ravel())
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
    #base_path = '/share/datasets/connectomics/connectomics_hp'
    base_path = '/home/npinto/datasets/connectomics/connectomics_hp'

    print 'dataset'
    ds = ConnectomicsHP(base_path)
    print 'meta'
    m = ds.meta()
    print 'ann'
    imgdl_ann = [e for e in m if 'annotation' in e]

    rf_size = (21, 21)
    #rf_size = (31, 31)
    lnorm_size = (11, 11)
    #lnorm_size = (5, 5)
    #n_filters = 2000#4 * np.prod(rf_size) + 100
    #n_filters = 1500
    n_filters = 1024
    #learning = 'randn'
    learning = 'imprint'
    # pe = 0.35354113395
    #learning = 'imprint_pos'
    # pe = 0.370370783012
    # kmeans like
    # pe = 0.371467231309
    DEBUG = False
    #DEBUG = True
    N_IMGS = 4

    print 'training image'
    #trn = imgdl_ann[-1]
    #trn_fname = trn['filename']
    trn_X_l = []
    trn_Y_l = []
    for i in range(N_IMGS):
        trn_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % i
        trn_X = (misc.imread(trn_fname, flatten=True) / 255.).astype('f')
        trn_X -= trn_X.mean()
        trn_X /= trn_X.std()
        #trn_Y = ds.get_annotation(trn['annotation'][0]).astype('f')
        trn_Y = (misc.imread(trn_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')
        trn_X_l += [trn_X]
        trn_Y_l += [trn_Y]
    trn_X = np.array(trn_X_l).reshape(N_IMGS*512, 512)
    trn_Y = np.array(trn_Y_l).reshape(N_IMGS*512, 512)

    print 'testing image'
    #tst = imgdl_ann[0]
    #tst_fname = tst['filename']
    tst_fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png'
    tst_X = (misc.imread(tst_fname, flatten=True) / 255.).astype('f')
    tst_X -= tst_X.mean()
    tst_X /= tst_X.std()
    #tst_Y = ds.get_annotation(tst['annotation'][0]).astype('f')
    tst_Y = (misc.imread(tst_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

    if DEBUG:
        trn_X = trn_X[:512, :512]
        trn_Y = trn_Y[:512, :512]
        tst_X = tst_X[:512, :512]
        tst_Y = tst_Y[:512, :512]
        #trn_X = trn_X[::4, ::4]
        #trn_Y = trn_Y[::4, ::4]
        #tst_X = tst_X[::4, ::4]
        #tst_Y = tst_Y[::4, ::4]

    #
    mdl1 = RPLogReg1(rf_size=rf_size,
                     lnorm_size=lnorm_size,
                     n_filters=n_filters,
                     learning=learning)
    start = time.time()
    mdl1.fit(trn_X, trn_Y.astype(bool))
    #trn_X1 = mdl1.predict(trn_X)[..., 0]

    #mdl2 = RPLogReg1(rf_size=rf_size,
                     #lnorm_size=lnorm_size,
                     #n_filters=n_filters)
    #mdl2.fit(trn_X1, trn_Y.astype(bool))

    #Y_pred = mdl2.predict(mdl1.predict(tst_X)[..., 0])
    Y_pred = mdl1.predict(tst_X)
    Y_true = tst_Y.copy()
    print 'pe =', pearson(Y_true.ravel(), Y_pred.ravel())
    end = time.time()

    print end-start

    from skimage import io
    io.use_plugin('freeimage')
    io.imsave('Y_pred.tif', Y_pred[..., 0], plugin='freeimage')
    io.imsave('Y_true.tif', Y_true, plugin='freeimage')

    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


if __name__ == '__main__':
    main()

