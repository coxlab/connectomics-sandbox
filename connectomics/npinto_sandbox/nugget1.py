import numpy as np
from scipy import misc
from scipy import ndimage as ndi
from scipy import io
from scipy.optimize import fmin_l_bfgs_b

from coxlabdata.connectomics_hp import ConnectomicsHP
from bangmetric.correlation import spearman, pearson
from bangmetric.precision_recall import average_precision
from skimage.util.shape import view_as_windows

import theano
from theano import tensor as T
from sthor.operation import lcdnorm3
from sthor.util.pad import filter_pad2d


DEFAULT_RF_SIZE = (21, 21)
DEFAULT_LNORM_SIZE = (11, 11)
DEFAULT_N_FILTERS = np.prod(DEFAULT_RF_SIZE)


class XXX(object):

    def __init__(self,
                 rf_size=DEFAULT_RF_SIZE,
                 lnorm_size=DEFAULT_LNORM_SIZE,
                 n_filters=DEFAULT_N_FILTERS,
                ):

        self.rf_size = rf_size
        self.lnorm_size = lnorm_size
        self.n_filters = n_filters

    def transform(self, X):

        rf = self.rf_size
        ns = self.lnorm_size
        #l1_nfilters = self.n_filters
        X_shape = X.shape

        X = filter_pad2d(np.atleast_3d(X), ns)
        X = lcdnorm3(X, ns)[..., 0]

        X = filter_pad2d(np.atleast_3d(X), rf)[..., 0]
        X = view_as_windows(X, rf)
        X = X.reshape(np.prod(X.shape[:2]), -1)

        print 'normalize...'
        X -= X.mean(0)
        X /= X.std(0)

        #print 'dot...'
        #fb = np.random.randn(X.shape[1], l1_nfilters).astype('f')
        #X = np.dot(X, fb).clip(0, np.inf)

        #print 'normalize...'
        #i -= i.mean(0)
        #i /= i.std(0)

        X = X.reshape(X_shape[:2] + (-1,))

        return X

    def fit(self, X, Y):

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
            Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, W, b)
            try:
                print 'ap', average_precision(Y_true.ravel(), Y_pred[:, 1].ravel())
                print 'pr', pearson(Y_true.ravel(), Y_pred[:, 1].ravel())
            except AssertionError:
                pass
            except ValueError:
                pass
            #print average_precision(a.ravel(), Y_pred.argmax(1).ravel())
            dloss = np.concatenate([dloss_W.ravel(), dloss_b.ravel()])
            return loss.astype('float64'), dloss.astype('float64')

        vars = np.concatenate([W.ravel(), b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(
            func,
            vars,
            iprint=1,
            factr=1e1,
            maxfun=10#00000
            )

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

    base_path = '/share/datasets/connectomics/connectomics_hp'

    ds = ConnectomicsHP(base_path)
    m = ds.meta()
    imgdl_ann = [e for e in m if 'annotation' in e]

    rf = (21, 21)
    #ns = (7, 7)
    #ns = (21, 21)
    ns = (11, 11)
    #rf = (5, 5)
    l1_nfilters = int(np.prod(rf)) + 100

    print imgdl_ann[0]

    a = gt = ds.get_annotation(imgdl_ann[0]['annotation'][0]).astype('f')
    i = misc.imread(imgdl_ann[0]['filename']) / 255.

    X = i.copy().astype('f')[::4, ::4]
    #X = i.copy().astype('f')
    Y = a.copy().astype('f')[::4, ::4]
    #Y = a.copy().astype('f')

    #
    mdl = XXX()
    mdl.fit(X, Y)

    Y_pred = mdl.predict(X)
    Y_true = Y.copy()
    print 'pr', pearson(Y_true.ravel(), Y_pred.ravel())

    #raise

    ##gt = ndi.gaussian_filter(a, 1).astype('f')
    #a = view_as_windows(a, rf)[int(ns[0]/2):-int(ns[0]/2), int(ns[1]/2):-int(ns[1]/2)]


    ##print ns[0]/2, ns[1]/2
    ##raise
    #a = a[:, :, rf[0]/2, rf[1]/2]
    #a = a.reshape(np.prod(a.shape[:2]), -1)

    #d = io.loadmat('/home/npinto/Dropbox/work/projects/connectomics/from_Verena/I00002_image_imProb.mat')
    #v = d['imProb']
    #print gt
    #print v
    ##v = v[int(rf[0]/2):-int(rf[0]/2), int(rf[1]/2):-int(rf[1]/2)]
    ##print v.shape, a.shape
    #print 'verena'
    #print 'ap', average_precision(gt.ravel(), v.ravel())
    #print 'pr', pearson(gt.ravel(), v.ravel())


    #i = lcdnorm3(np.atleast_3d(i), ns)[:, :, 0]
    #i = view_as_windows(i, rf)
    #i = i.reshape(np.prod(i.shape[:2]), -1)

    #print 'normalize...'
    #i -= i.mean(0)
    #i /= i.std(0)

    #print 'dot...'
    #i = np.dot(i, np.random.randn(i.shape[1], l1_nfilters).astype('f')).clip(0, np.inf)

    #print 'normalize...'
    #i -= i.mean(0)
    #i /= i.std(0)

    #print i.shape
    #print a.shape

    ##raise

    ##X = np.atleast_2d(i.ravel()).astype('float32').T
    #X = i.astype('float32')
    #print X.shape
    #Y_true = a.ravel().astype(long)

    ## -- initial variables
    #W = np.ones((X.shape[1], 2), dtype='float32')
    ##W = np.ones((X.shape[1], 1), dtype='float32')
    #W_size = W.size
    #W_shape = W.shape
    #b = np.zeros((2), dtype='float32')
    ##b = np.zeros((1), dtype='float32')

    ## -- theano program
    #_X = T.fmatrix()
    #_b = T.fvector()
    #_W = T.fmatrix()
    #_Y_true = T.lvector()

    #_Y_pred = T.nnet.softmax(T.dot(_X, _W) + _b)
    #_loss = -T.mean(T.log(_Y_pred)[T.arange(_Y_true.shape[0]), _Y_true])

    #_dloss_W = T.grad(_loss, _W)
    #_dloss_b = T.grad(_loss, _b)

    #_f = theano.function([_X, _W, _b],
                         #[_Y_pred],
                         #allow_input_downcast=True)

    #_f_df = theano.function([_X, _Y_true, _W, _b],
                            #[_Y_pred, _loss, _dloss_W, _dloss_b],
                            #allow_input_downcast=True)

    #def func(vars):
        ## unpack W and b
        #W = vars[:W_size].reshape(W_shape)
        #b = vars[W_size:]
        #Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, W, b)
        ##import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        ##print spearman(a.ravel(), Y_pred.argmax(1).ravel())
        ##import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        #try:
            #print 'ap', average_precision(a.ravel(), Y_pred[:, 1].ravel())
            #print 'pr', pearson(a.ravel(), Y_pred[:, 1].ravel())
            ##print average_precision(a.ravel(), Y_pred.ravel())
        #except AssertionError:
            #pass
        #except ValueError:
            #pass
        ##print average_precision(a.ravel(), Y_pred.argmax(1).ravel())
        #dloss = np.concatenate([dloss_W.ravel(), dloss_b.ravel()])
        #return loss.astype('float64'), dloss.astype('float64')

    #vars = np.concatenate([W.ravel(), b.ravel()])
    #from scipy.optimize import fmin_l_bfgs_b
    #best, bestval, info = fmin_l_bfgs_b(
        #func,
        #vars,
        ##approx_grad=True,
        #iprint=1,
        #factr=1e1,
        #maxfun=1000000
        ##factr=1e11,
        ##maxfun=1000
        #)
    ##print info
    ##print best.shape
    ##print bestval
    #vars = best
    #W = vars[:W_size].reshape(W_shape)
    #b = vars[W_size:]

    #Y_pred = _f(X, W, b)
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


if __name__ == '__main__':
    main()
