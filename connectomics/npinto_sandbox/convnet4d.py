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
from bangmetric.isbi12 import *

#l = (misc.lena() / 1.).astype('f')
print theano.config.openmp
theano.config.warn.sum_div_dimshuffle_bug = False

from xform import water

convnet_desc = [
    (8, 5, 2),
    (4, 4, 2),
    (8, 4, 2),
    (8, 4, 2),
    #(16, 7, 2),
    #(16, 7, 2),
    #(16, 7, 2),
    #(32, 5, 2),
    #(4, 5, 2),
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
    m=1e2,#1e3,
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
            #assert psize % 2 == 0, psize
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
        self.lr_min = 0.1#0.01
        self.total_n_batches = 0

        self._df_msize_best = None
        self._transform_msize_best = None

    def _safe_XY(self, X, Y=None):

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

    def _reshape_XY(self, X, Y=None, border='same'):

        assert border in ('same', 'valid')

        if Y is None:
            X = self._safe_XY(X)
        else:
            X, Y = self._safe_XY(X, Y)

        if X.ndim == 2:
            X = X.reshape(X.shape + (1,))

        X_d = X.shape[-1]
        fp = self.footprint
        pad_shape = fp/2, int(round(fp/2.) - 1)

        if border == 'same':
            # -- pad
            X = np.dstack([
                arraypad.pad(X[..., i], pad_shape, mode='reflect')
                for i in xrange(X_d)
            ])
        elif border == 'valid' and Y is not None:
            Y = Y[pad_shape[0]:-pad_shape[1],
                  pad_shape[0]:-pad_shape[1]]

        # -- rolling view
        X = view_as_windows(X, (fp, fp, 1))
        X = X.reshape(-1, fp, fp, X_d)

        # -- reshape to fit theano's convention
        X = np.ascontiguousarray(X.transpose(0, 3, 1, 2))

        if Y is None:
            return X
        else:
            assert len(X) == np.prod(Y.shape)
            return X, Y

    def transform(self, X, border='same', no_reshape=False, verbose=False):

        if no_reshape:
            Xrv = X
        else:
            assert X.shape[0] == X.shape[1], 'only square is implemented'
            Xrv = self._reshape_XY(X, border=border)

        Y_pred, self._transform_msize_best = theano_memory_hack(
            func_exp='self.f(*([slice_vars[0]] + self.fb_l + [self.W])).ravel()',
            local_vars=locals(),
            input_exps=['Xrv'],
            msize_best=self._transform_msize_best,
            )

        oh = ow = np.sqrt(len(Xrv))
        Y_pred = np.concatenate([elt for elt in Y_pred]).reshape(oh, ow)
        assert Y_pred.dtype == X.dtype

        return Y_pred

    def partial_fit(self, X, Y_true, border='valid', n_batches=100, batch_size=61*61):

        print '>>> X.shape:', X.shape
        print '>>> Y_true.shape:', Y_true.shape

        X, Y_true = self._safe_XY(X, Y_true)
        X, Y_true = self._reshape_XY(X, Y_true, border=border)
        Y_true = Y_true.ravel()

        print '>>> X.shape (new):', X.shape
        print '>>> Y_true.shape (new):', Y_true.shape

        ## -- filterbank renormalization
        print 'SELF SELF SELF'
        print type(self.fb_l)
        if self.fb_l is None:
            fb_l = []
        else:
            #fb_l = [fb / np.linalg.norm(fb) for fb in self.fb_l]
            fb_l = self.fb_l

        # -- theano stuff
        t_X = tensor.ftensor4()
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
            else:
                fb = fb_l[i]

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
        sigmoid_factor = 1#4
        t_Y_pred = 1. / (1. + tensor.exp(-sigmoid_factor * t_Y_pred))

        # -- loss
        epsilon = 0#.1
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
            #theano.printing.debugprint(f)
            self.f = f
            print 'df...'
            df = theano.function(
                [t_X] + t_fb_l + [t_W, t_Y_true],
                [t_loss] + t_dloss_dfb_l + [t_dloss_dW, t_Y_pred],
                )
            #theano.printing.debugprint(df)
            self.df = df
        else:
            f = self.f
            df = self.df

        if self.W is None:
            W = np.zeros((W_size + 1), dtype='float32')
        else:
            #W = self.W / np.linalg.norm(self.W)
            W = self.W

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
            current_fb_l, current_W = unpack_params(params)

            fb_norms = [np.linalg.norm(fb.ravel()) for fb in current_fb_l]
            W_norm = np.linalg.norm(current_W)

            # -- get loss and gradients from theano function
            # using the theano_memory_hack (sorry about that...)
            local_vars = locals()
            out, self._df_msize_best = theano_memory_hack(
                'self.df(*([slice_vars[0]] + current_fb_l + [current_W, slice_vars[1]]))',
                local_vars=local_vars,
                input_exps=('current_X', 'current_Y_true'),
                msize_best=self._df_msize_best,
                )
            loss = np.mean([o[0] for o in out])
            grads = []
            for gi in xrange(len(current_fb_l) + 1):  # filterbanks and W
                grads += [np.sum([o[gi + 1] for o in out], axis=0)]
            current_Y_pred = np.concatenate([o[-1].ravel() for o in out])

            # -- pack parameters
            grad_norms = [np.linalg.norm(g.ravel()) for g in grads]
            grads = np.concatenate([g.ravel() for g in grads])

            # -- early stopping / heuristic-based regularization
            th_norm = np.inf
            stop = False
            if (np.array(fb_norms) > th_norm).any() or W_norm > th_norm:
                print
                print '!!! Zeroing gradients / loss:'
                grads *= 0
                loss *= 0
                stop = True

            if self._n_calls > 0:
                try:
                    current_pe = pearson(current_Y_true, current_Y_pred)
                except AssertionError:
                    current_pe = 0
                #print 'current_pe:', current_pe
                if current_pe > 0.8:#9:
                    grads *= 0
                    loss *= 0

            if stop or (self._n_calls > 0 and self._n_calls % 10 == 0):
                print '#calls:', self._n_calls
                print 'current_X.shape', current_X.shape
                print 'current_Y_true.shape', current_Y_true.shape
                print 'elapsed:', time.time() - self.time_start
                print 'fb norms:', fb_norms
                print 'W norm:', W_norm
                print 'grad norms:', grad_norms
                if self._n_calls > 0:
                    current_pe = pearson(current_Y_true, current_Y_pred)
                    print 'current_pe:', current_pe
                #print '-' * 80
                #print '=' * 80

            self._n_calls += 1

            # fmin_l_bfgs_b needs double precision...
            return loss.astype('float64'), grads.astype('float64')

        # -- optimize
        lbfgs_params = DEFAULT_LBFGS_PARAMS
        #lbfgs_params['iprint'] = -1

        pos = (Y_true>0)
        neg = ~pos
        pos_ratio = pos.mean()
        neg_ratio = neg.mean()
        pos_size = int(batch_size*pos_ratio)
        neg_size = batch_size - pos_size

        for bi in xrange(n_batches):
            print "Batch %d" % (bi + 1)
            #if bi > 0:
            fb_l = [fb / max(np.linalg.norm(fb), 1) for fb in self.fb_l]
            W /= max(np.linalg.norm(self.W), 1)
            #self.W /= np.linalg.norm(self.W)

            #if self.total_n_batches > 0:
            if False:
                Y_pred = self.transform(X, no_reshape=True)
                pos_losses = (Y_true.flat[pos] - Y_pred.flat[pos]) ** 2.
                neg_losses = (Y_true.flat[neg] - Y_pred.flat[neg]) ** 2.
                pos_idx = (-pos_losses).argsort()[:pos_size]
                neg_idx = (-neg_losses).argsort()[:neg_size]
                #pos_losses = (pos_ratio - Y_pred.flat[pos]) ** 2.
                #neg_losses = (neg_ratio - Y_pred.flat[neg]) ** 2.
                #pos_idx = (pos_losses).argsort()[:pos_size]
                #neg_idx = (neg_losses).argsort()[:neg_size]
            else:
                ridx = self.rng.permutation(len(X))
                pos_idx = ridx[pos][:pos_size]
                neg_idx = ridx[neg][:neg_size]

            current_X = np.vstack([X[pos_idx], X[neg_idx]]).copy()
            current_Y_true = np.concatenate([Y_true[pos_idx], Y_true[neg_idx]]).copy()
            print 'RATIO RATIO', (current_Y_true > 0).mean(), pos_ratio, len(current_Y_true)
            self._n_calls = 0

            # pack parameters
            params = np.concatenate([fb.ravel() for fb in fb_l] + [W.ravel()])
            best, bestval, info = fmin_l_bfgs_b(
                minimize_me, 
                params,
                args=(current_X, current_Y_true),
                **lbfgs_params)
            #print info

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

            #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

            self.total_n_batches += 1

        return self


def main():

    trn_X, trn_Y, val_X, val_Y, tst_X, tst_Y = get_X_Y()
    #trn_X, trn_Y, tst_X, tst_Y, val_X, val_Y = get_X_Y()

    trn_X_orig = trn_X.copy()
    trn_Y_orig = trn_Y.copy()
    val_X_orig = val_X.copy()
    val_Y_orig = val_Y.copy()
    tst_X_orig = tst_X.copy()
    tst_Y_orig = tst_Y.copy()

    #tst_X_pad = arraypad.pad(tst_X, 512, mode='symmetric')
    #tst_Y_pad = arraypad.pad(tst_Y, 512, mode='symmetric')
    tst_X_pad = tst_X.copy()
    tst_Y_pad = tst_Y.copy()

    rng = np.random.RandomState(42)
    m = SharpMind(convnet_desc, rng=rng)

    batch_size = 10000#1e5#5000#1000#10000#61**2#1e3
    batch_size_factor = 1.1#1.2
    batch_size_max = 1e5
    n_batches = 1#0
    N_ITERATIONS = int(1e6)

    for iter in xrange(N_ITERATIONS):
        print '=' * 80
        print 'Iteration %d' % (iter + 1)
        print '-' * 80

        trn_X = trn_X_orig.copy()
        trn_Y = trn_Y_orig.copy()
        tst_X = tst_X_orig.copy()
        tst_Y = tst_Y_orig.copy()

        if iter > 0 and rng.binomial(1, .5):
            if rng.binomial(1, .5):
            #if True:
                print 'swirls...'
                trn_X, trn_Y = random_swirls(trn_X, trn_Y, rseed=iter)

            #if rng.binomial(1, .5):
            #if True:
                #print 'exp noise...'
                #trn_X = add_noise_experimental(trn_X)

            trn_X -= trn_X.mean()
            trn_X /= trn_X.std()

            if rng.binomial(1, .5):
                print 'rotate...'
                trn_X = ndimage.rotate(trn_X, iter * 90, prefilter=False, order=0)
                trn_Y = ndimage.rotate(trn_Y, iter * 90, prefilter=False, order=0)

            #if rng.binomial(1, .5):
                #print 'random xform...'
                #trn_X, trn_Y = get_random_transform(trn_X, trn_Y, rseed=iter)

            print('min=%.2f max=%.2f mean=%.2f std=%.2f'
                  % (trn_X.min(), trn_X.max(), trn_X.mean(), trn_X.std()))

        batch_size = min(int(batch_size_factor * batch_size), batch_size_max)
        m.lr_min = 0.01#1.0
        m.partial_fit(trn_X, trn_Y,
                      n_batches=n_batches, batch_size=batch_size)
        print 'pe...'
        #if iter == 0:
            #m.total_n_batches = 0

        start = time.time()
        tst_Y_true = gt = tst_Y_orig
        tst_Y_pred = gv = median_filter(m.transform(tst_X_orig), 2)
        tst_pe = pearson(gt.ravel(), gv.ravel())
        tst_rd = rand_error(gt, gv)
        val_pe = pearson(val_Y.ravel(), m.transform(val_X_orig).ravel())
        ttime = time.time() - start
        print
        print
        print 'FINAL VAL_PE (orig):', val_pe
        print 'FINAL TST_PE (orig):', tst_pe
        print 'FINAL TST_RD (orig):', tst_rd
        print 'time:', ttime
        print
        print
        print 'total_n_batches:', m.total_n_batches
        print

        print ">>> Saving Y_true.tif"
        tst_Y_true = tst_Y_true.astype('f')
        tst_Y_true -= tst_Y_true.min()
        tst_Y_true /= tst_Y_true.max()
        io.imsave('Y_true.tif', tst_Y_true, plugin='freeimage')
        misc.imsave('Y_true.tif.png', tst_Y_true)

        print ">>> Saving Y_pred..."
        fname = 'Y_pred.iter=%05d.%s.tif' % (iter, tst_rd)
        print fname
        tst_Y_pred -= tst_Y_pred.min()
        tst_Y_pred /= tst_Y_pred.max()

        io.imsave(fname, tst_Y_pred, plugin='freeimage')
        misc.imsave(fname + '.png', tst_Y_pred)

        #start = time.time()
        #rng2 = np.random.RandomState(34)
        #gt_l = None
        #gv_l = None
        #SIZE = 61 + fp
        ##pe_l = []
        #for ti in xrange(64):
            #j, i = rng2.randint(0, len(tst_X_pad)-SIZE+1, size=2)
            ##print j, i
            #tst_X2 = tst_X_pad[j:j+SIZE, i:i+SIZE].copy()
            #tst_Y2 = tst_Y_pad[j:j+SIZE, i:i+SIZE].copy()
            #gt = tst_Y2[pad_shape[0]:-pad_shape[1],
                        #pad_shape[0]:-pad_shape[1]]
            #gv = m.transform(tst_X2, border='valid')
            #gt = gt.ravel()
            #gv = gv.ravel()
            ##pe_l += [pearson(gt, gv)]
            #if gt_l is None:
                #gt_l = gt
                #gv_l = gv
            #else:
                #gt_l = np.concatenate([gt_l, gt])
                #gv_l = np.concatenate([gv_l, gv])
            ##print pearson(gt_l, gv_l), np.mean(pe_l), xxx

        #tst_pe = pearson(gt_l, gv_l)
        #ttime = time.time() - start
        #print
        #print
        #print 'FINAL TST_PE (patches):', tst_pe, ttime
        #print
        #print
        ##print '*' * 80

if __name__ == '__main__':
    main()
