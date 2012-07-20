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
import time

from os import path, environ
from scipy import ndimage
from skimage.filter import median_filter

HOME = environ.get("HOME")

from bangmetric import *

#l = (misc.lena() / 1.).astype('f')

from xform import water

convnet_desc = [
    (16, 5, 2),
    (16, 5, 2),
    (16, 5, 2),
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


def get_X_Y():
    N_IMGS = 1
    N_IMGS_VAL = 1
    print 'training image'
    trn_X_l = []
    trn_Y_l = []
    for i in range(N_IMGS):
        #i += 10
        trn_fname = path.join(
            HOME, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % i)
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
        val_fname = path.join(
            HOME, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-%02d.png' % k)
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
    tst_fname = path.join(HOME, 'datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png')
    print tst_fname
    tst_X = (misc.imread(tst_fname, flatten=True) / 255.).astype('f')
    #tst_X -= tst_X.min()
    #tst_X /= tst_X.max()
    tst_X -= tst_X.mean()
    tst_X /= tst_X.std()
    tst_Y = (misc.imread(tst_fname.replace('volume', 'labels'), flatten=True) > 0).astype('f')

    return trn_X, trn_Y, tst_X, tst_Y


class SharpMind(object):

    def __init__(self, convnet_desc):

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

        self.fb_l = None
        self.W = None
        #self.b = None

        self.f = None

    def transform_Y(self, Y):

        Y = Y.reshape((1, 1) + Y.shape)

        output_shape = Y.shape
        for i, (nf, fsize, psize) in enumerate(self.convnet_desc):
            input_shape = output_shape

            # hack to get the output_shape
            output_shape = np.empty(
                (input_shape[0], nf,
                 (input_shape[2] - fsize + 1),
                 (input_shape[3] - fsize + 1)),
                dtype='uint8',
                )[:, :, ::psize, ::psize].shape

            fsize2 = fsize // 2
            Y = Y[:, :, fsize2:-fsize2, fsize2:-fsize2]
            Y = Y[:, :, ::psize, ::psize]
            assert Y.shape[2:] == output_shape[2:], (Y.shape, output_shape)

        return Y[0, 0]

    def transform(self, X, Y_true=None):

        assert X.ndim == 2
        assert X.dtype == 'float32'

        # -- reshape to fit theano's convention
        #print '>>> X.shape:', X.shape
        X = X.reshape((1, 1) + X.shape)
        #print '>>> X.shape:', X.shape

        Y_pred = self.f(*([X] + self.fb_l + [self.W]))

        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')

        return Y_pred[0]

    def partial_fit(self, X, Y_true):

        assert X.ndim == 2
        assert X.dtype == 'float32'

        assert Y_true.ndim == 2
        assert Y_true.dtype == 'float32'

        # -- reshape to fit theano's convention
        print '>>> X.shape:', X.shape
        X = X.reshape((1, 1) + X.shape)
        print '>>> X.shape:', X.shape

        print '>>> Y_true.shape:', Y_true.shape
        Y_true = Y_true.reshape((1, 1) + Y_true.shape)
        print '>>> Y_true.shape:', Y_true.shape

        # -- theano stuff
        t_X = tensor.ftensor4()
        t_Y_true = tensor.ftensor4()
        t_W = tensor.fvector()

        if self.fb_l is None:
            fb_l = []
        else:
            fb_l = [fb / np.linalg.norm(fb) for fb in self.fb_l]

        t_input_l = []
        input_l = []
        t_fb_l = []
        t_f_l = []
        t_p_l = []
        t_output_l = []

        # -- ConvNet
        output_shape = X.shape
        t_output = t_X
        for i, (nf, fsize, psize) in enumerate(self.convnet_desc):
            input_shape = output_shape
            if self.fb_l is None:
                np.random.seed(42)
                #fb = np.random.uniform(size=(nf, input_shape[1], fsize, fsize)).astype('f')
                #fb_shape = (nf, input_shape[1], fsize, fsize)
                #fb = np.ones(fb_shape, dtype='f')
                fb = np.random.randn(nf, input_shape[1], fsize, fsize).astype('f')
                fb -= fb.mean()
                fb /= np.linalg.norm(fb.ravel())
                fb_l += [fb]

            t_input = t_output
            t_fb = tensor.ftensor4()
            #t_f = tensor.tanh(nnet.conv2d(t_input, t_fb))
            #t_f = tensor.maximum(0, nnet.conv2d(t_input, t_fb) ** 2.)
            #t_f = tensor.maximum(0, nnet.conv2d(t_input, t_fb)) ** 2.
            t_f = tensor.maximum(0, nnet.conv2d(t_input, t_fb))
            t_p = downsample.max_pool_2d(t_f, (psize, psize))
            t_output = t_p

            # hack to get the output_shape
            output_shape = np.empty(
                (input_shape[0], nf,
                 (input_shape[2] - fsize + 1),
                 (input_shape[3] - fsize + 1)),
                dtype='uint8',
                )[:, :, ::psize, ::psize].shape

            fsize2 = fsize // 2
            Y_true = Y_true[:, :, fsize2:-fsize2, fsize2:-fsize2]
            Y_true = Y_true[:, :, ::psize, ::psize]
            assert Y_true.shape[2:] == output_shape[2:], (Y_true.shape, output_shape)

            input_l += [input]
            t_input_l += [t_input]
            t_fb_l += [t_fb]
            t_f_l += [t_f]
            t_p_l += [t_p]
            t_output_l += [t_output]

        # -- MLP
        W_size = fb_l[-1].shape[0]
        t_Y_pred = tensor.tensordot(t_output, t_W[:-1], axes=[(1,), (0,)]) + t_W[-1]
        #t_Y_pred = nnet.sigmoid(t_Y_pred)
        sigmoid_factor = 4#2
        t_Y_pred = 1. / (1. + tensor.exp(-sigmoid_factor*t_Y_pred))
        #t_Y_pred = (1. + t_Y_pred) / 2

        #t_Y_true = t_Y_true[:, 0, :, :]
        #t_Y_true = 2. * (t_Y_true > 0.) - 1
        #t_loss = ((t_Y_pred - t_Y_true[:, 0, :, :]) ** 2.).mean()
        epsilon = 0.1#.1#2#.1#.2#1#2#3#0.2
        l2_regularization = 0#1e-4#3#0#1e-6#3#.1
        #t_Y_true = 2. * (t_Y_true > 0.) - 1
        #t_loss = (tensor.maximum(0, 1 - t_Y_pred*t_Y_true[:, 0, :, :] - epsilon) ** 2.).mean()
        t_loss = (tensor.maximum(0, (t_Y_pred - t_Y_true[:, 0, :, :]) ** 2 - epsilon) ** 2.).mean()
        for t_fb in t_fb_l:
            t_loss += l2_regularization * tensor.dot(t_fb.flatten(), t_fb.flatten())
        t_loss += l2_regularization * tensor.dot(t_W.flatten(), t_W.flatten())

        t_dloss_dfb_l = [tensor.grad(t_loss, t_fb) for t_fb in t_fb_l]
        t_dloss_dW = tensor.grad(t_loss, t_W)
        #t_dloss_db = tensor.grad(t_loss, t_W)

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

        #ciw = self.ciw
        def minimize_me(params):

            stdout.write('.')
            stdout.flush()

            # unpack parameters
            fb_l, W = unpack_params(params)
            self.fb_l = fb_l
            self.W = W
            #fb_l = [fb / np.linalg.norm(fb.ravel()) for fb in fb_l]
            #W_norm = np.linalg.norm(W.ravel())
            #if W_norm > 0:
                #W = W / W_norm
            #self.fb_l = fb_l
            #self.W = W
            #new_fb_l = []
            #for fb in fb_l:
                #fb_norm = np.linalg.norm(fb.ravel())
                #if fb_norm > 2:
                    #print 'renorm', fb.shape
                    #fb /= fb_norm
                #new_fb_l.append(fb)
            fb_norms = [np.linalg.norm(fb.ravel()) for fb in fb_l]
            W_norm = np.linalg.norm(W)
            #for fb, fb_norm in zip(fb_l, fb_no
            #if (np.array(fb_norms) > 1.5).any():

            # get loss and gradients from theano function
            out = df(*([X] + fb_l + [W, Y_true]))
            loss = out[0]
            grads = out[1:]

            # pack parameters
            grad_norms = [np.linalg.norm(g.ravel()) for g in grads]
            grads = np.concatenate([g.ravel() for g in grads])
            th_norm = 2#1.5
            if (np.array(fb_norms) > th_norm).any() or W_norm > th_norm:
                grads[:] = 0
            #if self._n_calls >= 100:#200:
                #grads[:] = 0

            if self._n_calls % 10 == 0:
                print
                print '=' * 80
                print '#calls:', self._n_calls
                print 'elapsed:', time.time() - self.time_start
                print 'fb norms:', fb_norms
                print 'W norm:', W_norm
                print 'grad norms:', grad_norms
                print '-' * 80
                try:
                    print 'trn_shape:', self.transform_Y(self.trn_X).shape
                    trn_pe = pearson(self.transform_Y(self.trn_Y).ravel(),
                                     self.transform(self.trn_X).ravel())
                    print 'trn_pe:', trn_pe
                    print 'tst_shape:', self.transform_Y(self.tst_X).shape
                    tst_pe = pearson(self.transform_Y(self.tst_Y).ravel(),
                                     self.transform(self.tst_X).ravel())
                    print 'tst_pe:', tst_pe
                    #ratio = tst_pe / trn_pe
                    #print 'ratio:', ratio
                    #if ratio < 0.8:#9:
                        #grads[:] = 0
                except AssertionError:
                    pass
                print '=' * 80

            self._n_calls += 1
            # fmin_l_bfgs_b needs double precision...
            return loss.astype('float64'), grads.astype('float64')

        # pack parameters
        lbfgs_params = DEFAULT_LBFGS_PARAMS
        params = np.concatenate([fb.ravel() for fb in fb_l] + [W.ravel()])
        best, bestval, info = fmin_l_bfgs_b(minimize_me, params, **lbfgs_params)

        #lr = 1e-3
        ##ssa = 1. / (i + 1)
        ##xa = (1 - ssa) * xa + ssa * x
        #EPSILON = 1e-6
        ##Xorig = X.copy()
        ##footprint = self.footprint
        ##rng = np.random.RandomState(42)
        ##print footprint
        ##print len(Xorig)
        #while True:
            ##j, i = rng.randint(0, Xorig.shape[-1]-footprint, size=2)
            ##X = Xorig[j:j+footprint, i:i+footprint]
            ##print X.shape
            #try:
                #l, g = minimize_me(params)
                #params -= lr * g
                ##print 'max', max(abs(g)), abs(params).max()
                ##print max(g), max(params)
                ##th = lr * max(abs(g / params))
                ###th = max(abs(g) / abs(params)) 
                ##print 'th:', th
                ##if th > 0.05:
                    ##lr *= 1.05
                ##else:
                    ##lr *= 0.95
                ##print lr
                #if (abs(g) < EPSILON).all():
                    #break
            #except KeyboardInterrupt:
                #pass

        #fb_l, W = unpack_params(params)
        #self.fb_l = fb_l
        #self.W = W

        #best, bestval, info = fmin_l_bfgs_b(minimize_me, params, **lbfgs_params)

        best_fb_l, best_W = unpack_params(best)
        self.fb_l = best_fb_l
        self.W = best_W
        #self.b = best_b

        return self


def main():

    trn_X, trn_Y, tst_X, tst_Y = get_X_Y()

    m = SharpMind(convnet_desc)
    #pad = m.footprint
    #print pad

    ##trn_X_pad = arraypad.pad(trn_X, 512, mode='symmetric')
    ##trn_Y_pad = arraypad.pad(trn_Y, 512, mode='symmetric')
    ##tst_X_pad = arraypad.pad(tst_X, 512, mode='symmetric')
    ##tst_Y_pad = arraypad.pad(tst_Y, 512, mode='symmetric')
##
    ##trn_X = trn_X_pad
    ##trn_Y = trn_Y_pad
    ##tst_X = tst_X_pad
    ##tst_Y = tst_Y_pad
    #m.trn_Y = trn_Y
    #m.trn_X = trn_X
    #m.tst_Y = tst_Y
    #m.tst_X = tst_X
    #print '*' * 80
    #print 'trn final shape:', m.transform_Y(trn_Y).shape
    #print 'tst final shape:', m.transform_Y(tst_Y).shape
    #print '*' * 80
    ##m.ciw = [4, 2, 1]
    #m.partial_fit(trn_X, trn_Y)
    ##m.ci = [0]
    ##m.partial_fit(trn_X, trn_Y)
    ##m.ci = [1]
    ##m.partial_fit(trn_X, trn_Y)
    ##m.ci = [2]
    ##m.partial_fit(trn_X, trn_Y)

    #tst_pe = pearson(m.transform_Y(tst_Y).ravel(), m.transform(tst_X).ravel())
    #print 'tst_pe', tst_pe


    #raise

    trn_X_pad_orig = arraypad.pad(trn_X, 512, mode='symmetric')
    trn_Y_pad_orig = arraypad.pad(trn_Y, 512, mode='symmetric')
    #misc.imsave('trn_X_pad.png', trn_X_pad)
    #misc.imsave('trn_Y_pad.png', trn_Y_pad)
    tst_X_pad = arraypad.pad(tst_X, 512, mode='symmetric')
    tst_Y_pad = arraypad.pad(tst_Y, 512, mode='symmetric')

    SIZE = 512#*2#*2#3*512-1#1024
    N_BAGS = 1000
    FOLLOW_AVG = False
    #DECAY = 1e-3


    rng = np.random.RandomState(42)
    fb_l = None
    W = None
    lr_min = 1e-1#2# = 0.1#2#1e-2#1#1e-3#1#1e-2
    for bag in xrange(N_BAGS):
        #m = SharpMind(convnet_desc)
        #SIZE = (b + 1) * SIZE
        #trn_X_pad = water(trn_X_pad, sigma=10)
        #misc.imsave('trn_X_pad.png', trn_X_pad)
        #trn_X_pad = trn_X_pad_orig
        #if bag > 0:
            #trn_X_pad = water(trn_X_pad, sigma=0.8)
            #misc.imsave('trn_X_pad.png', trn_X_pad)
        angle = rng.randint(0, 360 + 1)
        trn_X_pad = ndimage.rotate(trn_X_pad_orig, angle,
                                   prefilter=False, order=0, mode='reflect')
        trn_Y_pad = ndimage.rotate(trn_Y_pad_orig, angle,
                                   prefilter=False, order=0, mode='reflect')
        #trn_X_pad = water(trn_X_pad, sigma=1)
        misc.imsave('trn_X_pad.png', trn_X_pad)

        print '>>> Finding a balanced patch...'
        bal = 1
        bal_th = 0.8
        while bal > bal_th:
            j, i = rng.randint(0, len(trn_X_pad)-SIZE+1, size=2)
            #print j, i
            trn_X = trn_X_pad[j:j+SIZE, i:i+SIZE].copy()
            trn_Y = trn_Y_pad[j:j+SIZE, i:i+SIZE].copy()
            pos = m.transform_Y(trn_Y)>0
            #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
            #print pos
            bal = 1. * pos.sum() / pos.size
        print 'bal:', bal, j, i

        m.trn_Y = trn_Y
        m.trn_X = trn_X
        m.tst_Y = tst_Y
        m.tst_X = tst_X
        m.partial_fit(trn_X, trn_Y)
        if fb_l is None:
            fb_l = m.fb_l
            W = m.W
        else:
            lr = 1. / (bag + 1)
            lr = np.maximum(lr, lr_min)
            #lr = lr_min
            print 'lr:', lr
            fb_l = [(1 - lr) * fb_l[i] + lr * m.fb_l[i] for i in xrange(len(fb_l))]
            #fb_l = [fb / np.linalg.norm(fb) for fb in fb_l]
            W = lr * m.W + (1 - lr) * W
            #W /= np.linalg.norm(W)

        if not FOLLOW_AVG:
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
            gt = m.transform_Y(tst_Y2).ravel()
            gv = m.transform(tst_X2).ravel()
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
        print 'tst_pe:', tst_pe
        print
        print
        print '*' * 80
        if not FOLLOW_AVG:
            m.fb_l = fb_l_bak
            m.W = W_bak

        if tst_pe > 0.743:
            raise

    #trn_Y = m.transform_Y(trn_Y)
    #tst_pe = pearson(m.transform_Y(tst_Y).ravel(), m.transform(tst_X).ravel())
    #print 'tst_pe', tst_pe


if __name__ == '__main__':
    main()
