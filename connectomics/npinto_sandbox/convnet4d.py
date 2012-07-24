from sys import stdout
from scipy.optimize import fmin_l_bfgs_b
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
from skimage.util import view_as_blocks
import time

from os import path, environ
from scipy import ndimage
from skimage.filter import median_filter
from random_connectomics import *
from connectomics_noise import *
from connectomics_swirls import *

from skimage import io
io.use_plugin('freeimage')

HOME = environ.get("HOME")

from bangmetric import *

#l = (misc.lena() / 1.).astype('f')
print theano.config.openmp
theano.config.warn.sum_div_dimshuffle_bug = False

from xform import water

convnet_desc = [
    (16, 5, 2),
    (16, 5, 2),
    (16, 5, 2),
    #(16, 5, 2),
    #(16, 5, 2),
    #(48, 3, 2),
    #(2, 5, 2),
]

#mlp_desc = [
    #200,
#]

DEFAULT_LBFGS_PARAMS = dict(
    iprint=10,
    #factr=1e7,
    factr=1e7,#12,#10,#7,#$12,#7,
    maxfun=1e4,
    m=1000,
    )


from connectomics_data import get_X_Y

from theano_hacks import theano_memory_hack


class SharpMind(object):

    def __init__(self, convnet_desc, rng=None):

        # -- model description
        for i, (nf, fsize, psize) in enumerate(convnet_desc):
            print(">>> Layer %d: nf=%d, fsize=%d, psize=%d"
                  % ((i + 1), nf, fsize, psize))
            assert fsize % 1 == 0, fsize
            assert psize % 2 == 0, psize
        self.convnet_desc = convnet_desc

        # -- footprint / padding
        footprint = 1
        for _, fsize, psize in convnet_desc[::-1]:
            footprint *= psize
            footprint += fsize - 1
        assert footprint > 0, footprint
        self.footprint = footprint
        print ">>> Footprint: %d" % footprint

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.fb_l = None
        self.W = None
        #self.b = None

        self.f = None

        self.msize_best = None

        self.lr_exp = 0
        self.eta0 = 1
        self.lr_min = 0.01
        self.total_n_batches = 0

    def _safe_X_Y(self, X, Y=None):

        assert X.ndim == 2 or X.ndim == 3
        assert X.dtype == 'float32'
        X_h, X_w = X.shape[:2]

        if Y is None:
            return X

        else:
            assert Y.ndim == 2
            assert Y.dtype == 'float32'
            Y_h, Y_w = Y.shape

            assert Y_h == X_h
            assert Y_w == X_w

            return X, Y

    def _reshape_X(self, X):

        X = self._safe_X_Y(X)

        if X.ndim == 2:
            X = X.reshape(X.shape + (1,))

        X_d = X.shape[-1]

        # -- pad
        fp = self.footprint
        pad_shape = fp/2, int(round(fp/2.) - 1)
        X = np.dstack([
            arraypad.pad(X[..., i], pad_shape, mode='reflect')
            for i in xrange(X_d)
        ])

        # -- rolling view
        X = view_as_windows(X, (fp, fp, 1))
        X = X.reshape(-1, fp, fp, X_d)

        # -- reshape to fit theano's convention
        X = X.swapaxes(1, 3)

        return X

    def transform(self, X, verbose=False):

        Xrv = self._reshape_X(X)

        Y_pred, msize_best = theano_memory_hack(
            func_exp='self.f(*([Xin] + self.fb_l + [self.W])).ravel()',
            local_vars=locals(),
            input_exp='Xrv',
            slice_exp='Xin',
            )

        Y_pred = Y_pred.reshape(X.shape[:2])
        assert Y_pred.dtype == X.dtype

        return Y_pred

    def partial_fit(self, X, Y_true, n_batches=100, batch_size=61*61):

        print '>>> X.shape:', X.shape
        print '>>> Y_true.shape:', Y_true.shape

        X, Y_true = self._safe_X_Y(X, Y_true)
        X = self._reshape_X(X)
        Y_true = Y_true.ravel()

        print '>>> X.shape (new):', X.shape
        print '>>> Y_true.shape (new):', Y_true.shape

        # -- filterbank renormalization
        if self.fb_l is None:
            fb_l = []
        else:
            fb_l = [fb / np.linalg.norm(fb) for fb in self.fb_l]

        # -- theano stuff
        t_X = tensor.ftensor4()
        #t_Y_true = tensor.ftensor4()
        t_Y_true = tensor.fvector()
        t_W = tensor.fvector()

        t_input_l = []
        input_l = []
        t_fb_l = []
        t_F_l = []
        t_A_l = []
        t_P_l = []
        t_output_l = []

        # -- ConvNet
        t_output = t_X
        prev_nf = X.shape[1]
        for i, (nf, fsize, psize) in enumerate(self.convnet_desc):
            # init filterbanks if needed
            if self.fb_l is None:
                np.random.seed(42)
                fb = np.random.randn(nf, prev_nf, fsize, fsize).astype('f')
                fb -= fb.mean()
                fb /= np.linalg.norm(fb.ravel())
                fb_l += [fb]

            t_input = t_output

            #t_input -= tensor.mean(t_input)
            #t_input /= tensor.std(t_input)

            # -- filter
            t_fb = tensor.ftensor4()
            t_F = nnet.conv2d(t_input, t_fb,
                              #image_shape=input_shape,
                              filter_shape=fb.shape
                             )

            # -- activ
            #t_A = tensor.tanh(t_F)
            t_A = tensor.maximum(t_F, 0)
            #t_A = tensor.clip(t_F, 0, 1)

            # -- pool
            t_P = downsample.max_pool_2d(t_A, (psize, psize))
            t_output = t_P

            input_l += [input]
            t_input_l += [t_input]
            t_fb_l += [t_fb]
            t_F_l += [t_F]
            t_A_l += [t_A]
            t_P_l += [t_P]
            t_output_l += [t_output]

            prev_nf = nf

        t_output = tensor.flatten(t_output, outdim=2)

        #t_output -= tensor.mean(t_output)
        #t_output /= tensor.std(t_output)

        # -- multi-layer perceptron (mlp)
        W_size = fb_l[-1].shape[0]
        t_Y_pred = tensor.dot(t_output, t_W[:-1]) + t_W[-1]
        sigmoid_factor = 4
        t_Y_pred = 1. / (1. + tensor.exp(-sigmoid_factor * t_Y_pred))

        # -- loss
        epsilon = 0.1
        l2_regularization = 0
        t_loss = tensor.maximum(0, ((t_Y_pred - t_Y_true) ** 2.) - epsilon)
        t_loss = t_loss ** 2.
        t_loss = t_loss.mean()

        # regularization
        for t_fb in t_fb_l:
            t_loss += l2_regularization * tensor.dot(t_fb.flatten(), t_fb.flatten())
        t_loss += l2_regularization * tensor.dot(t_W.flatten(), t_W.flatten())

        # -- gradients
        t_dloss_dfb_l = [tensor.grad(t_loss, t_fb) for t_fb in t_fb_l]
        t_dloss_dW = tensor.grad(t_loss, t_W)

        # -- compile functions
        if self.f is None:
            print '>>> Compiling theano functions...'
            print 'f...'
            f = theano.function([t_X] + t_fb_l + [t_W], t_Y_pred)
            self.f = f
            print 'df...'
            df = theano.function(
                [t_X] + t_fb_l + [t_W, t_Y_true],
                [t_loss] + t_dloss_dfb_l + [t_dloss_dW],
                )
            self.df = df
        else:
            f = self.f
            df = self.df

        if self.W is None:
            W = np.zeros((W_size + 1), dtype='float32')
        else:
            W = self.W / np.linalg.norm(self.W)

        self.fb_l = fb_l
        self.W = W
        self._n_calls = 0

        self.time_start = time.time()

        def unpack_params(params):
            params = params.astype('f')
            fb_l_new = []
            i = 0
            for fb in self.fb_l:
                fb_new = params[i:i+fb.size].reshape(fb.shape)
                fb_l_new.append(fb_new)
                i += fb.size
            fb_l = fb_l_new

            W = params[i:i+self.W.size].reshape(self.W.shape)
            i += self.W.size

            return fb_l, W

        def minimize_me(params, *args):

            current_X = args[0]
            current_Y_true = args[1]

            stdout.write('.')
            stdout.flush()

            # unpack parameters
            fb_l, W = unpack_params(params)
            self.fb_l = fb_l
            self.W = W

            fb_norms = [np.linalg.norm(fb.ravel()) for fb in fb_l]
            W_norm = np.linalg.norm(W)

            # get loss and gradients from theano function
            out = df(*([current_X] + fb_l + [W, current_Y_true]))
            loss = out[0]
            grads = out[1:]

            # pack parameters
            grad_norms = [np.linalg.norm(g.ravel()) for g in grads]
            grads = np.concatenate([g.ravel() for g in grads])
            th_norm = 2
            stop = False
            if (np.array(fb_norms) > th_norm).any() or W_norm > th_norm:
                print
                print '!!! Zeroing gradients / loss:'
                grads *= 0
                loss *= 0
                stop = True

            if stop or (self._n_calls > 0 and self._n_calls % 10 == 0):
                print 'current_X.shape', current_X.shape
                print '#calls:', self._n_calls
                print 'elapsed:', time.time() - self.time_start
                print 'fb norms:', fb_norms
                print 'W norm:', W_norm
                print 'grad norms:', grad_norms
                #print '-' * 80
                #print '=' * 80

            self._n_calls += 1
            # fmin_l_bfgs_b needs double precision...
            return loss.astype('float64'), grads.astype('float64')

        # -- optimize
        lbfgs_params = DEFAULT_LBFGS_PARAMS
        lbfgs_params['iprint'] = -1

        # pack parameters
        params = np.concatenate([fb.ravel() for fb in fb_l] + [W.ravel()])

        for bi in xrange(n_batches):
            print "Batch %d" % (bi + 1)
            if bi > 0:
                self.fb_l = [fb / max(np.linalg.norm(fb), 1) for fb in self.fb_l]
                self.W /= max(np.linalg.norm(self.W), 1)

            ridx = self.rng.permutation(len(X))[:batch_size]
            print len(ridx)
            current_X = X[ridx].copy()
            current_Y_true = Y_true[ridx].copy()
            self._n_calls = 0
            best, bestval, info = fmin_l_bfgs_b(
                minimize_me, 
                params,
                args=(current_X, current_Y_true),
                **lbfgs_params)
            print info

            best_fb_l, best_W = unpack_params(best)

            if self.lr_exp > 0:
                lr = 1. * self.eta0 / (1. + self.eta0 * self.total_n_batches) ** self.lr_exp
            else:
                lr = 1. / (1. + self.total_n_batches)
            lr = np.maximum(lr, self.lr_min)

            print 'lr:', lr
            self.fb_l = [lr * best_fb_l[i] + (1 - lr) * self.fb_l[i]
                         for i in xrange(len(fb_l))]

            self.W = lr * best_W + (1 - lr) * self.W
            self.total_n_batches += 1

        return self


def main():

    trn_X, trn_Y, tst_X, tst_Y = get_X_Y()

    m = SharpMind(convnet_desc)

    #m.trn_Y = trn_Y
    #m.trn_X = trn_X
    #m.tst_Y = tst_Y
    #m.tst_X = tst_X
    maxsize = None
    N_ITERATIONS = 10
    for iter in xrange(N_ITERATIONS):
        print '=' * 80
        print 'Iteration %d' % (iter + 1)
        print '-' * 80
        m.partial_fit(trn_X[:maxsize, :maxsize], trn_Y[:maxsize, :maxsize],
                      n_batches=1, batch_size=61**2)
        print 'pe...'
        trn_pe = pearson(trn_Y[:128, :128].ravel(), m.transform(trn_X[:128, :128]).ravel())
        print 'trn_pe:', trn_pe
        tst_pe = pearson(tst_Y[:128, :128].ravel(), m.transform(tst_X[:128, :128]).ravel())
        print 'tst_pe:', tst_pe

        print 'total_n_batches:', m.total_n_batches
        print '=' * 80

    return

    trn_X_orig = trn_X.copy()
    trn_Y_orig = trn_Y.copy()
    #tst_X_orig = tst_X.copy()
    #tst_Y_orig = tst_Y.copy()

    #trn_X_pad_orig = arraypad.pad(trn_X, 512, mode='symmetric')
    #trn_Y_pad_orig = arraypad.pad(trn_Y, 512, mode='symmetric')
    tst_X_pad = arraypad.pad(tst_X, 512, mode='symmetric')
    tst_Y_pad = arraypad.pad(tst_Y, 512, mode='symmetric')

    SIZE = 512#*2#*2#3*512-1#1024
    N_BAGS = 10000
    FOLLOW_AVG = 10#True#False
    #DECAY = 1e-3

    start = time.time()

    rng = np.random.RandomState(42)
    fb_l = None
    W = None
    lr_exp = 0#0.75
    eta0 = 1#2#1#2#1#.2#1#.2#0.8#1#.5#1#0.1
    lr_min = 0.01#1e-3#0.1#1e-3#0.05#1#25#01#05#1#01#05#5e-2
    #gaussian_sigma = 1#0.5


    for bag in xrange(N_BAGS):
        print "BAGGING ITERATION", (bag + 1)
        ##m = SharpMind(convnet_desc)
        ##SIZE = (b + 1) * SIZE
        ##trn_X_pad = water(trn_X_pad, sigma=10)
        ##misc.imsave('trn_X_pad.png', trn_X_pad)
        ##trn_X_pad = trn_X_pad_orig
        ##if bag > 0:
            ##trn_X_pad = water(trn_X_pad, sigma=0.8)
            ##misc.imsave('trn_X_pad.png', trn_X_pad)
        #angle = rng.randint(0, 360 + 1)
        #trn_X_pad = ndimage.rotate(trn_X_pad_orig, angle,
                                   #prefilter=False, order=0, mode='reflect')
        #trn_Y_pad = ndimage.rotate(trn_Y_pad_orig, angle,
                                   #prefilter=False, order=0, mode='reflect')
        ##trn_X_pad = water(trn_X_pad, sigma=1)
        #misc.imsave('trn_X_pad.png', trn_X_pad)

        #print '>>> Finding a balanced patch...'
        #bal = 0#np.inf#1
        #bal_th = (trn_Y>0).mean()
        #bal_tol = 0.01
        #print 'bal_th:', bal_th
        #print 'bal_tol:', bal_tol
        #while abs(1 - bal / bal_th) > bal_tol:
            #j, i = rng.randint(0, len(trn_X_pad)-SIZE+1, size=2)
            ##print j, i
            #trn_X = trn_X_pad[j:j+SIZE, i:i+SIZE].copy()
            #trn_Y = trn_Y_pad[j:j+SIZE, i:i+SIZE].copy()
            #pos = m.transform_Y(trn_Y)>0
            ##import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
            ##print pos
            #bal = 1. * pos.sum() / pos.size
            ##print abs(1 - bal / bal_th), bal_tol
        #print 'bal:', bal, j, i
        trn_X = trn_X_orig.copy()
        trn_Y = trn_Y_orig.copy()

        #if rng.binomial(1, .5):
            #print 'flip h...'
            #trn_X = trn_X[:, ::-1]
            #trn_Y = trn_Y[:, ::-1]

        #if rng.binomial(1, .5):
            #print 'flip v...'
            #trn_X = trn_X[::-1, :]
            #trn_Y = trn_Y[::-1, :]

        #if rng.binomial(1, .5):
        if bag > 0:
            if True:
                print 'swirls...'
                trn_X, trn_Y = random_swirls(trn_X, trn_Y, rseed=bag)

            #if rng.binomial(1, .5):
            #if True:
                #print 'exp noise...'
                #trn_X = add_noise_experimental(trn_X)

            trn_X -= trn_X.mean()
            trn_X /= trn_X.std()

            #if rng.binomial(1, .5):
            if True:
                print 'rotate...'
                trn_X = ndimage.rotate(trn_X, bag * 90, prefilter=False, order=0)
                trn_Y = ndimage.rotate(trn_Y, bag * 90, prefilter=False, order=0)

            #if rng.binomial(1, .5):
            if True:
                print 'random xform...'
                trn_X, trn_Y = get_random_transform(trn_X, trn_Y, rseed=bag)

            print trn_X.min(), trn_X.max()

        #gaussian_sigma = rng.uniform(0, .5)
        #print 'gaussian_sigma:', gaussian_sigma
        #trn_X = ndimage.gaussian_filter(trn_X, gaussian_sigma)

        m.trn_Y = trn_Y
        m.trn_X = trn_X
        m.tst_Y = tst_Y
        m.tst_X = tst_X
        m.partial_fit(trn_X, trn_Y)

        fp = m.footprint

        X3d = view_as_windows(
            arraypad.pad(tst_X, (fp/2, int(round(fp/2.) - 1)), mode='reflect'),
            (fp, fp)
            ).reshape(-1, fp, fp) 

        if fb_l is None:
            fb_l = m.fb_l
            W = m.W
        else:
            if lr_exp > 0:
                lr = 1. * eta0 / (1. + eta0 * bag) ** lr_exp
            else:
                lr = 1. / (1. + bag)
            lr = np.maximum(lr, lr_min)
            #lr = lr_min
            print 'lr:', lr
            fb_l = [(1 - lr) * fb_l[i] + lr * m.fb_l[i] for i in xrange(len(fb_l))]
            #fb_l = [fb / np.linalg.norm(fb) for fb in fb_l]
            W = lr * m.W + (1 - lr) * W
            #W /= np.linalg.norm(W)

        #if not FOLLOW_AVG:
        fb_l_bak = deepcopy(m.fb_l)
        W_bak = deepcopy(m.W)
        m.fb_l = fb_l
        m.W = W
        trn_Y = m.transform_Y(trn_Y)
        print 'tst_Y.shape:', tst_Y.shape
        print '*' * 80
        #tst_pe_l = []
        rng2 = np.random.RandomState(34)
        gt_l = None
        gv_l = None
        for ti in xrange(64):
            j, i = rng2.randint(0, len(tst_X_pad)-SIZE+1, size=2)
            #print j, i
            tst_X2 = tst_X_pad[j:j+SIZE, i:i+SIZE].copy()
            tst_Y2 = tst_Y_pad[j:j+SIZE, i:i+SIZE].copy()
            gt = m.transform_Y(tst_Y2)
            gv = m.transform(tst_X2)
            misc.imsave('gt_%02d.png' % ti, gt)
            misc.imsave('gv_%02d.png' % ti, gv)
            gt = gt.ravel()
            gv = gv.ravel()
            #print gv
            #print gv.shape
            #gv = median_filter(gv, radius=2).ravel()
            if gt_l is None:
                gt_l = gt
                gv_l = gv
            else:
                gt_l = np.concatenate([gt_l, gt])
                gv_l = np.concatenate([gv_l, gv])
            #tst_pe = pearson(m.transform_Y(tst_Y2).ravel(), m.transform(tst_X2).ravel())
            #print j, i#, '%d tst_pe: %s' % (ti, tst_pe)
            #tst_pe_l += [tst_pe]
        #print 'mean:', np.mean(tst_pe_l)
        tst_pe = pearson(gt, gv)
        print
        print
        print 'FINAL TST_PE:', tst_pe
        print
        print

        print ">>> Computing tst_Y_pred..."
        tic = time.time()
        tst_Y_pred = m.transform3d(X3d).reshape(tst_X.shape).astype('float32')
        toc = time.time()
        print tst_Y_pred.shape
        print 'time:', toc - tic

        print ">>> Saving Y_true.tif"
        tst_Y = tst_Y.astype('f')
        tst_Y -= tst_Y.min()
        tst_Y /= tst_Y.max()
        io.imsave('Y_true.tif', tst_Y, plugin='freeimage')
        misc.imsave('Y_true.tif.png', tst_Y)

        print ">>> Saving Y_pred..."
        tst_Y_pred -= tst_Y_pred.min()
        tst_Y_pred /= tst_Y_pred.max()
        tst_pe_full = pearson(tst_Y.ravel(), tst_Y_pred.ravel())

        print 'FINAL TST_PE_FULL:', tst_pe_full
        fname = 'Y_pred.bag%05d.%s.tif' % (bag, tst_pe_full)
        print fname

        io.imsave(fname, tst_Y_pred, plugin='freeimage')
        misc.imsave(fname + '.png', tst_Y_pred)


        print '*' * 80
        #if not FOLLOW_AVG or bag % FOLLOW_AVG > 0:
        if not FOLLOW_AVG:
            m.fb_l = fb_l_bak
            m.W = W_bak
        else:
            print "FOLLOW AVG !!!!!"

        #if tst_pe > 0.76435:
            #end = time.time()
            #print 'time:', end - start
            #return

if __name__ == '__main__':
    main()
