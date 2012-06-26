# IPython log file

import theano
from theano import tensor
t_X = tensor.fmatrix()
t_w = tensor.fvector()
f = theano.function([t_x, t_w], tensor.dot(t_X, t_w.T))
f = theano.function([t_X, t_w], tensor.dot(t_X, t_w.T))
f = theano.function([t_X, t_w], tensor.dot(t_X, t_w))
X = np.random.randn(10, 4)
w = np.random.randn(4)
f(X, w)
f = theano.function([t_X, t_w], tensor.dot(t_X, t_w.T))
f(X, w)
f = theano.function([t_X, t_w], tensor.dot(t_X, t_w.T), allow_input_downcast=True)
f(X, w)
f = theano.function([t_X, t_w], tensor.dot(t_X, t_w), allow_input_downcast=True)
f(X, w)
f = theano.function([t_X, t_w], theano.nnet.sigmoid(tensor.dot(t_X, t_w)), allow_input_downcast=True)
from theano import nnet
from theano.sandbox import nnet
f = theano.function([t_X, t_w], tensor.nnet.sigmoid(tensor.dot(t_X, t_w)), allow_input_downcast=True)
f(X, w)
y = 2 * np.random.randn(len(X)) > 0 - 1
y
y = 2. * (np.random.randn(len(X)) > 0.) - 1
y
f = theano.function([t_X, t_w], 2. * tensor.nnet.sigmoid(tensor.dot(t_X, t_w)) - 1, allow_input_downcast=True)
f(X, w)
H = 2. * tensor.nnet.sigmoid(tensor.dot(t_X, t_w)) - 1
t_H = 2. * tensor.nnet.sigmoid(tensor.dot(t_X, t_w)) - 1
t_loss = tensor.maximum(0, 1 - H - m)
m = 0.2
t_loss = tensor.maximum(0, 1 - H - m)
t_y = tensor.fvector()
t_M = t_y * H
t_loss = tensor.maximum(0, 1 - t_M - m)
t_loss = tensor.mean(tensor.maximum(0, 1 - t_M - m) ** 2.)
t_dloss_dw = tensor.grad(t_loss, t_w)
_f_df = theano.function([t_x, t_y, t_w], [t_loss, t_dloss_dw], allow_input_downcast=True)
_f_df = theano.function([t_X, t_y, t_w], [t_loss, t_dloss_dw], allow_input_downcast=True)
def fun(params):
    w = params.astype('f')
    c, d = _f_df(X, y, w)
    return c.astype('d'), d.astype('d')
from scipy.optimize import *
fmin_l_bfgs_b(fun, np.zeros(X.shape[1]))
fmin_l_bfgs_b(fun, np.zeros(X.shape[1]), iprint=1)
best = fmin_l_bfgs_b(fun, np.zeros(X.shape[1]), iprint=1)
best.shape
best = fmin_l_bfgs_b(fun, np.zeros(X.shape[1]), iprint=1)[0]
best.shape
f(X, best)
y
M = f(X, best) * y
np.maximum(0, 1 - M -m)
get_ipython().magic(u'logstart ')
