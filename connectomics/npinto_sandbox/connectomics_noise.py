from scipy import ndimage
from scipy import misc
import numpy as np

import connectomics_data as cd


def get_mask(shape, p=0.999, sigma=10, th=0.2):
    r = np.random.binomial(1, p, size=shape).astype('f')
    r2 = ndimage.gaussian_filter(r, sigma)
    r2 -= r2.min()
    r2 /= r2.max()
    r2 = (r2 - th).clip(0, 1)
    r2 -= r2.min()
    r2 /= r2.max()
    return (1 - r2)

def get_poi(shape, lam=1.0, sigma=1.0):
    p = np.random.poisson(1, size=shape).astype('f')
    p = ndimage.gaussian_filter(p, 1)
    p -= p.min()
    p /= p.max()
    return p

def get_gau(X, sigma=1.0):
    g = ndimage.gaussian_filter(X, 1)
    return g


def add_noise_experimental(X, rseed=None):

    if rseed is not None:
        np.random.seed(rseed)

    Y = 0 * X

    gau = get_gau(X, sigma=1)
    mask = get_mask(X.shape, p=0.9, sigma=10., th=0.8)
    Y += (mask * gau + (1. - mask) * X)

    gau = get_gau(X, sigma=10.0)
    mask = get_mask(X.shape, p=0.999, sigma=10., th=0.5)
    Y += 1e-1 * (mask * gau + (1. - mask) * X)

    poi = get_poi(X.shape, lam=1.0, sigma=1.0)
    mask = get_mask(X.shape, p=0.999, sigma=10., th=0.2)
    Y += mask * poi + (1. - mask) * X

    return Y


def main():
    X = cd.get_X_Y()[0]

    for i in xrange(10):

        Y = add_noise_experimental(X, rseed=i)

        fname = 'noise_%02d.png' % i
        print fname
        misc.imsave(fname, Y)

if __name__ == '__main__':
    main()
