import matplotlib
matplotlib.use('Agg')

from os import path
import skimage.io as io
from scipy.ndimage import rotate

# -- path to the ISBI PNG images
base_path = "/home/poilvert/connectomics/isbi_dataset/"


def get_images(trn_img_idx=[0, 1, 2], tst_img_idx=[29], rotate_img=False,
               use_true_tst_img=False):
    """
    Extracts and normalized the required ISBI dataset images.

    Parameters
    ----------

    ``trn_img_idx``: list
        list of integers giving the indices of the ISBI training images to
        extract

    ``tst_img_idx``: list
        list of integers giving the indices of the ISBI testing images to
        extract

    ``rotate_img``: bool
        if True, the code will generate a 90, 180 and 270 degres rotated image
        for each training image

    ``use_true_tst_img``: bool
        if True, the "true" ISBI **testing** images are used. If False, the
        testing images are extracted from the training set. The reason for this
        keyword is that for model selection, one needs testing images with
        ground truth annotations in order to compute metrics values. But only
        the images from the training set have ground truths

    Returns
    -------

    a list of lists. Each list looks like [trn_X_l, trn_Y_l, tst_X_l, tst_Y_l],
    where ``trn_X_l`` is a list of 2D training images (as numpy arrays),
    ``trn_Y_l`` is a list of ground truth annotations (also 2D numpy arrays),
    ``tst_X_l`` is a list of 2D testing images, and ``tst_Y_l`` is (possibly) a
    list of ground truth annotations (it can also be empty, in which case it
    means that there is no annotations for the testing images)
    """

    io.use_plugin('pil')

    # -- extract training images (and ground truth labels)
    trn_X_l, trn_Y_l = [], []

    print 'training images'
    for idx in trn_img_idx:

        print ' image %3i' % idx

        fname = path.join(base_path, 'train-volume.tif-%02d.png' % idx)

        trn_X = io.imread(fname, as_grey=True).astype('f')
        trn_X -= trn_X.mean()
        trn_X /= trn_X.std()

        trn_Y = (io.imread(fname.replace('volume', 'labels'), as_grey=True) > 0.).astype('f')

        if rotate_img:
            trn_X_l += [trn_X, rotate(trn_X, 90), rotate(trn_X, 180), rotate(trn_X, 270)]
            trn_Y_l += [trn_Y, rotate(trn_Y, 90), rotate(trn_Y, 180), rotate(trn_Y, 270)]
        else:
            trn_X_l += [trn_X]
            trn_Y_l += [trn_Y]

    # -- extract testing images (and ground truth labels)
    tst_X_l, tst_Y_l = [], []

    print 'testing images'
    for idx in tst_img_idx:

        print ' image %3i' % idx

        if use_true_tst_img:
            fname = path.join(base_path, 'test-volume.tif-%02d.png' % idx)
        else:
            fname = path.join(base_path, 'train-volume.tif-%02d.png' % idx)

        tst_X = io.imread(fname, as_grey=True).astype('f')
        tst_X -= tst_X.mean()
        tst_X /= tst_X.std()

        if not use_true_tst_img:
            tst_Y = (io.imread(fname.replace('volume', 'labels'), as_grey=True) > 0.).astype('f')
            tst_Y_l += [tst_Y]

        tst_X_l += [tst_X]

    # -- return training and testing lists. The reasons for the double sum
    # comes from the fact that we may want to have many cross validation folds
    # for the training and testing sets
    return [[trn_X_l, trn_Y_l, tst_X_l, tst_Y_l]]
