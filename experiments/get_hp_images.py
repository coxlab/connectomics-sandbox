#!/usr/env/python

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>

# license: BSD

"""
This program exposes a very limited view of Hans Pfister connectomics
dataset.
The dataset consists of 10 1024*1024 training images that will be split
into a total of 40 512*512 images.

By default, the annotation images will be slightly dilated from the
original training images to ressemble more the ISBI annotations.

Don't forget to export the environment variable ::

    CONNECTOMICS_HP_BASE_PATH

to the appropriate path, in order for the program to find Hans Pfister
dataset on disk.
"""

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np
import skimage.io as io
from coxlabdata import ConnectomicsHP
from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_windows
from sklearn import cross_validation

DTYPE=np.float32
NORMALIZE=True
DILATE=True
dilation=2.
height=512
width=512
EPSILON=1e-5


# -- utility functions
def pad2d(arr_in, filter_shape2d, constant=-1, reverse_padding=False):
    """Returns a padded array with constant values for the padding. The
    first two dimensions of the input array are padded, not the third
    one.

    Parameters
    ----------

    `arr_in`: array-like, shape = [height, width]
        input 2D array to pad

    `filter_shape2d`: 2-tuple
        size of the "filter" from which we decide how many units on both
        sides of the array needs to be added

    `constant`: float
        value for the padding

    `reverse_padding`: bool
        in this case the array is not padded but rather the aprons are
        removed directly from the array

    Returns
    -------
    `arr_out`: array-like
        padded input array
    """

    assert arr_in.ndim == 2
    assert len(filter_shape2d) == 2

    fh, fw = np.array(filter_shape2d, dtype=int)
    inh, inw = arr_in.shape

    if fh % 2 == 0:
        h_left, h_right = fh / 2, fh / 2 - 1
    else:
        h_left, h_right = fh / 2, fh / 2
    if fw % 2 == 0:
        w_left, w_right = fw / 2, fw / 2 - 1
    else:
        w_left, w_right = fw / 2, fw / 2

    # -- dimensions of the output array
    if reverse_padding:
        h_new = inh - h_left - h_right
        w_new = inw - w_left - w_right
    else:
        h_new = h_left + inh + h_right
        w_new = w_left + inw + w_right

    assert h_new >= 1
    assert w_new >= 1

    # -- makes sure the padding value is of the same type
    #    as the input array elements
    arr_out = np.empty((h_new, w_new), dtype=arr_in.dtype)
    arr_out[:] = constant

    if reverse_padding:
        arr_out[:] = arr_in[h_left:inh - h_right,
                            w_left:inw - w_right]

    else:
        arr_out[h_left:h_new - h_right,
                w_left:w_new - w_right] = arr_in

    return arr_out


def dilate_annotation(img, cutoff=1.):
    """
    `img` is a boolean or integer image. Every element strictly above zero
    will be set to 1 and the ones below or equal to zero will be set to 0.

    `cutoff` corresponds to a critical radius that gives the maximum distance
    from a "ground truth pixel" at which to accept a pixel to belong to the
    new "ground truth". In short it gives the amount of dilation around every
    ground truth pixel."
    """

    img = np.asarray(img)
    assert img.ndim == 2

    epsilon = 1e-5

    cutoff = float(cutoff)
    cutoffi = int(cutoff)
    img = (img > 0)
    h, w = img.shape[:2]

    size = 2 * cutoffi + 1
    x, y = np.mgrid[-cutoffi:cutoffi:complex(size),
                    -cutoffi:cutoffi:complex(size)]
    mask_filter = np.where(np.sqrt(x ** 2 + y ** 2) <= cutoff + epsilon,
                           1., 0.).astype(np.bool)

    mask = pad2d(img, (size, size), constant=False, reverse_padding=False)
    mask = view_as_windows(mask, (size, size)).reshape(-1, size * size)
    out = np.dot(mask, mask_filter.ravel())
    mask = (out > 0).reshape(h, w)

    return mask


# -- core function to extract images
def get_images(trn_img_idx=[0, 1, 2], tst_img_idx=[29]):
    """
    This function extract training and testing images from the raw training
    images in Hans Pfister dataset, which originally consists of 10 1024*1024
    tif images.

    Parameters
    ----------

    ``trn_img_idx``: list
        this gives the indices of the selected training images

    ``tst_img_idx``: list
        this gives the indices of the selected testing images

    Returns
    -------

    the code returns a list of lists. The format is ::

        trn_X_l, trn_Y_l, tst_X_l, tst_Y_l

    where ``trn_X_l`` is the list of training images, ``trn_Y_l`` is the list
    of ground truth images for the training set, and the same goes for the
    testing set.
    The reason for the "list of lists" is that each list in the output is
    considered to be a cross-validation fold.

    Limitations:
    ------------

    The indices should be chosen between 0 and 39 (there are 10 images in the
    original training set, that are then split into 4 512*512 sub-images, this
    leads to 40 images in total)

    Another limitation is the fact that this code generates only **one** fold
    of cross-validation. It is the responsibility of the user to extend this
    code if more folds of cross validation are expected.
    """

    # -- get all training images from dataset
    data = ConnectomicsHP()
    metadata = data.meta()
    annotated_metadata = [imgd for imgd in metadata if 'annotation' in imgd]

    log.info('extracting raw images from dataset')
    raw_imgs = [io.imread(imgd['filename'], as_grey=True).astype(DTYPE)
                for imgd in annotated_metadata]

    assert len(raw_imgs) > 0, 'could not find any training images'

    # -- get all annotation images from dataset
    log.info('extracting annotation images from dataset')
    gt_raw_imgs = [data.get_annotation(imgd['annotation'][0]).astype(DTYPE)
                   for imgd in annotated_metadata]

    # -- we dilate the annotations to make the boundaries thicker than just one
    # pixel (this is on purpose to look more like the ISBI annotations)
    if DILATE:
        log.info('requested annotation dilation with factor %5.3f' %
                 dilation)
        new_gt_raw_imgs = []
        for img in gt_raw_imgs:
            new_gt_raw_imgs += [dilate_annotation(img, cutoff=dilation)]
        gt_raw_imgs = new_gt_raw_imgs

    assert len(raw_imgs) == len(gt_raw_imgs)

    # -- we break down all the 1024*1024 images into 512*512 images
    ref_height, ref_width = raw_imgs[0].shape[:2]

    assert height <= ref_height
    assert width <= ref_width

    n_blk_h = ref_height / height
    n_blk_w = ref_width / width

    new_height = n_blk_h * height
    new_width = n_blk_w * width

    expected_n_imgs = n_blk_h * n_blk_w * len(raw_imgs)

    arr = np.empty((expected_n_imgs, height, width), dtype=DTYPE)
    gt_arr = np.empty((expected_n_imgs, height, width), dtype=DTYPE)

    batch_size = n_blk_h * n_blk_w

    for idx, (img, gt_img) in enumerate(zip(raw_imgs, gt_raw_imgs)):

        crop = img[0:new_height, 0:new_width]
        gt_crop = gt_img[0:new_height, 0:new_width]
        blk_view = view_as_blocks(crop,
                                 (height, width)).reshape(-1,
                                               height, width)
        gt_blk_view = view_as_blocks(gt_crop,
                                     (height, width)).reshape(-1,
                                               height, width)

        arr[idx*batch_size:(idx+1)*batch_size, ...] = blk_view.copy()
        gt_arr[idx*batch_size:(idx+1)*batch_size, ...] = gt_blk_view.copy()

    # -- we select only the required images for training/testing
    final_trn_imgs = [arr[i, ...] for i in trn_img_idx]
    final_tst_imgs = [arr[i, ...] for i in tst_img_idx]

    # -- normalization pixel-wise (zero mean, unit variance)
    if NORMALIZE:
        log.info('normalizing images')
        dummy = []
        for img in final_trn_imgs:
            mean = img.mean()
            std = img.std()
            if std < EPSILON:
                log.warn('one of the images is almost a constant')
                log.warn('is it expected ?')
                std = 1.
            dummy += [(img - mean) / std]
        final_trn_imgs = dummy
        dummy = []
        for img in final_tst_imgs:
            mean = img.mean()
            std = img.std()
            if std < EPSILON:
                log.warn('one of the images is almost a constant')
                log.warn('is it expected ?')
                std = 1.
            dummy += [(img - mean) / std]
        final_tst_imgs = dummy

    # -- we make sure that the annotation images are in the same format as the
    # ones from the ISBI challenge (i.e. "True" indicates the interior of a
    # cell, while "False" indicates a membrane)
    final_trn_gt_imgs = [(gt_arr[i, ...] <= 0.).astype(DTYPE) for i in trn_img_idx]
    final_tst_gt_imgs = [(gt_arr[i, ...] <= 0.).astype(DTYPE) for i in tst_img_idx]

    return [[final_trn_imgs, final_trn_gt_imgs,
             final_tst_imgs, final_tst_gt_imgs]]
