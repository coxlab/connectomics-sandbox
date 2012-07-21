import numpy as np
from scipy import misc

from skimage.transform import swirl

import connectomics_data as cd


def random_swirls(X, Y, n_swirls=1, order=0, rseed=None):

    X2 = X.copy()
    X2 -= X2.min()
    X2 /= X2.max()

    Y2 = Y.copy()
    Y2 -= Y2.min()
    Y2 /= Y2.max()

    rng = np.random.RandomState(rseed)

    for i in xrange(n_swirls):

        yc = rng.randint(X.shape[0])
        xc = rng.randint(X.shape[1])
        center = (yc, xc)

        radius = rng.uniform(1, np.max(X.shape))

        strength = rng.uniform(0.1, 0.8)

        X2 = swirl(X2, rotation=0, mode='mirror', order=order,
                    center=center, strength=strength, radius=radius)
        Y2 = swirl(Y2, rotation=0, mode='mirror', order=order,
                    center=center, strength=strength, radius=radius)

    X2 = X2.astype(X.dtype)
    Y2 = (Y2 > 0.5).astype(Y.dtype)

    return X2, Y2

def main():
    X, Y = cd.get_X_Y()[:2]

    for i in xrange(10):
        X2, Y2 = random_swirls(X, Y, n_swirls=10, rseed=i)

        out = np.hstack((X2, Y2))

        fname = 'swirl_%02d.png' % i
        print fname
        misc.imsave(fname, out)

        #fname = 'swirl_%02d_X.png' % i
        #print fname
        #misc.imsave(fname, X2)

        #fname = 'swirl_%02d_Y.png' % i
        #print fname
        #misc.imsave(fname, Y2)


if __name__ == '__main__':
    main()
