import numpy as np
from scipy import misc

from skimage import transform

import random
from random import choice, randint

def get_random_transform(X, Y, rseed=42):

    random.seed(rseed)

    angle_degrees = randint(0, 360)

    # hacky grid random parameters
    x_shift = randint(0, X.shape[1] // 8) * choice([-1., +1.])
    y_shift = randint(0, X.shape[0] // 8) * choice([-1., +1.])
    x_scale = 1. + choice([0, .1, .2])
    y_scale = 1. + choice([0, .1, .2])
    x_skew = choice([0, 1e-4, 1e-5, 1e-6, 1e-7])
    y_skew = choice([0, 1e-4, 1e-5, 1e-6, 1e-7])

    # build projection matrices
    angle_radians = np.radians(angle_degrees)
    sin_angle, cos_angle = np.sin(angle_radians), np.cos(angle_radians)

    xform = np.array([
        [cos_angle, -sin_angle, x_shift],
        [sin_angle,  cos_angle, y_shift],
        [0, 0, 1]
    ])

    height, width = X.shape
    y_center, x_center = height / 2., width / 2.
    center = np.array([
        [1, 0, -x_center],
        [0, 1, -y_center],
        [0, 0, 1]
    ])

    skew = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [x_skew, y_skew, 1]]
    )

    # final homography projection matrix
    H = skew.dot(np.linalg.inv(center).dot(xform).dot(center))

    Xout = transform.fast_homography(X, H, mode='mirror')
    Xout = Xout.astype(X.dtype)

    Yout = transform.fast_homography(Y, H, mode='mirror')
    Yout = (Yout > 0.5).astype(Y.dtype)

    return Xout, Yout


def main():

    import connectomics_data
    X, Y = connectomics_data.get_X_Y()[:2]

    for i in xrange(100):
        print i
        Xout, Yout = get_random_transform(X, Y, rseed=i)
        misc.imsave('img2_%02d_X.png' % i, Xout)
        misc.imsave('img2_%02d_Y.png' % i, Yout)

if __name__ == '__main__':
    main()
