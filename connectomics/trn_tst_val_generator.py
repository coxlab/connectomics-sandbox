#!/usr/bin/env python

"""
This program takes the connectomics dataset and prepares
a set of train/test and validation images for the screening
task.
"""

# -- imports
import numpy as np
from coxlabdata.connectome import ConnectomicsHP
from parameters import DATASET_PATH
from scipy.misc import imread
from scipy.ndimage import gaussian_filter
from skimage.shape import view_as_blocks

# -- default dtype for all images and annotations
DTYPE = np.float32

# -- default image indices for train/test and val sets
DEFAULT_TRN_TST_IMG_Z_IDX = [71]
DEFAULT_VAL_IMG_Z_IDX = [3]

# -- whether we use normalized images (zero mean, unit variance)
NORMALIZE = True

# -- defaults for ground truth Gaussian blurring
SIGMA = 1.
EPSILON = 0.2


def generate(divide_factor=2,
             trn_tst_img_z_idx=DEFAULT_TRN_TST_IMG_Z_IDX,
             val_img_z_idx=DEFAULT_VAL_IMG_Z_IDX
             ):

    # -----------------------
    # -- Get dataset metadata
    # -----------------------

    obj = ConnectomicsHP(DATASET_PATH)
    metadata = obj.meta()
    annotated_metadata = [imgd for imgd in metadata if 'annotation' in imgd]

    #--------------------------------------
    # -- Extract raw images and annotations
    # -------------------------------------

    # -- select trn/tst and val images
    trn_tst_raw_imgs = [imread(imgd['filename'], flatten=True).astype(DTYPE)
                        for imgd in annotated_metadata if imgd['z'] in
                        trn_tst_img_z_idx
                        ]
    val_raw_imgs = [imread(imgd['filename'], flatten=True).astype(DTYPE)
                    for imgd in annotated_metadata if imgd['z'] in
                    val_img_z_idx
                    ]

    # -- corresponding Ground Truth annotations
    trn_tst_gt_raw_imgs = \
            [obj.get_annotation(imgd['annotation'][0]).astype(DTYPE)
                for imgd in annotated_metadata if imgd['z'] in
                trn_tst_img_z_idx
                ]
    val_gt_raw_imgs = \
            [obj.get_annotation(imgd['annotation'][0]).astype(DTYPE)
                for imgd in annotated_metadata if imgd['z'] in
                val_img_z_idx
                ]

    # ----------------------------------------------------
    # -- Operations on images and annotations (normalizing
    #    images and blurring annotations)
    # ----------------------------------------------------

    # -- we may also want to normalize the images
    if NORMALIZE:
        trn_tst_imgs = []
        for img in trn_tst_raw_imgs:
            new_img = (img - img.mean()) / img.std()
            trn_tst_imgs += [new_img]
        val_imgs = []
        for img in val_raw_imgs:
            new_img = (img - img.mean()) / img.std()
            val_imgs += [new_img]
    else:
        trn_tst_imgs = trn_tst_raw_imgs
        val_imgs = val_raw_imgs

    # -- we want annotations to be {-1., +1.} with a slight
    #    Gaussian smear (to help a little the classifier)
    trn_tst_gt_imgs = []
    for img in trn_tst_gt_raw_imgs:
        new_img = np.where(img <= 0., -1., 1.)
        new_img = gaussian_filter(new_img, SIGMA)
        new_img = np.where(new_img > -1. + EPSILON, 1., -1.)
        trn_tst_gt_imgs += [new_img]
    val_gt_imgs = []
    for img in val_gt_raw_imgs:
        new_img = np.where(img <= 0., -1., 1.)
        new_img = gaussian_filter(new_img, SIGMA)
        new_img = np.where(new_img > -1. + EPSILON, 1., -1.)
        val_gt_imgs += [new_img]

    assert len(trn_tst_imgs) == len(trn_tst_gt_imgs)
    assert len(val_imgs) == len(val_gt_imgs)

    # -------------------------------------
    # -- Creating the cross validation sets
    # -------------------------------------

    # -- images original size
    h, w = trn_tst_imgs[0].shape[:2]

    # -- size of final images after we split the original
    #    images
    h_new, w_new = int(np.round(float(h) / float(divide_factor))), \
                   int(np.round(float(w) / float(divide_factor)))
    while (int(divide_factor) * h_new > h) or (int(divide_factor) * w_new > w):
        h_new -= 1
        w_new -= 1
    h_tot = int(divide_factor) * h_new
    w_tot = int(divide_factor) * w_new

    # -- loop over cross validation folds to build the train/test set
    trn_tst_ll = []
    for i in xrange(int(divide_factor)):
        for j in xrange(int(divide_factor)):

            # -- list to contain all the training/testing and gt
            #    images for the specified cross validation fold
            trn_l = []
            tst_l = []
            trn_gt_l = []
            tst_gt_l = []

            # -- list of training/testing images
            for img, gt in zip(trn_tst_imgs, trn_tst_gt_imgs):
                blk_view = view_as_blocks(img[:h_tot, :w_tot], (h_new, w_new))
                blk_view_gt = view_as_blocks(gt[:h_tot, :w_tot], (h_new, w_new))
                tst_l += [blk_view[i, j, :, :]]
                tst_gt_l += [blk_view_gt[i, j, :, :]]
                for l in xrange(int(divide_factor)):
                    for p in xrange(int(divide_factor)):
                        if (l != i) or (p != j):
                            trn_l += [blk_view[l, p, :, :]]
                            trn_gt_l = [blk_view_gt[l, p, :, :]]

            # -- append new lists of train/test images to
            #    ``trn_tst_ll``
            trn_tst_ll.append([trn_l, trn_gt_l, tst_l, tst_gt_l])

    # -- build the validation set
    val_l = []
    val_gt_l = []
    for img, gt in zip(val_imgs, val_gt_imgs):
        blk_view = view_as_blocks(img[:h_tot, :w_tot], (h_new, w_new))
        blk_view_gt = view_as_blocks(gt[:h_tot, :w_tot], (h_new, w_new))
        for i in xrange(int(divide_factor)):
            for j in xrange(int(divide_factor)):
                val_l += [blk_view[i, j, :, :]]
                val_gt_l += [blk_view_gt[i, j, :, :]]
    val_ll = [val_l, val_gt_l]

    sets = dict(trn_tst=trn_tst_ll, val=val_ll)

    return sets
