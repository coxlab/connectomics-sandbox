#!/usr/bin/env python

"""
This program will:
    1. create a cross validated screening task with
       train/validation and testing sets
    2. trains a classifier for every cross validation
       fold and computes the performance for both the
       validation images and testing images
    3. computes the metrics
"""

# -- for logging
import logging as log
log.basicConfig(level=log.INFO)

import numpy as np
from numpy.random import permutation
from trn_tst_val_generator import generate
from sthor.model.slm_new import SequentialLayeredModel
from skimage.shape import view_as_windows
import genson
from parameters import PLOS09
from asgd import NaiveBinaryASGD as Classifier
from util import OnlineScaler
from util import get_reduced_tm
from metrics import ap
from random import shuffle

from pymongo import Connection

connection = Connection()
db = connection['connectome']
posts = db.posts

# -----------
# -- defaults
# -----------

DEFAULT_TRN_VAL_IMG_Z_IDX = [71]
DEFAULT_TST_IMG_Z_IDX = [3]
DEFAULT_GENSON_SEED = 1
RANDOMIZE = True
NEPOCH = 1
NBH = 1
NBW = 1

# -------------------------------------------------
# -- generate training, validation and testing sets
# -------------------------------------------------

log.info('generating cross validation sets')
sets_dictionnary = generate(divide_factor=2,
                            trn_val_img_z_idx=DEFAULT_TRN_VAL_IMG_Z_IDX,
                            tst_img_z_idx=DEFAULT_TST_IMG_Z_IDX)

trn_val_ll = sets_dictionnary['trn_val']
tst_l = sets_dictionnary['tst']

# ------------------------------
# -- generation of the SLM model
# ------------------------------

while True:

    log.info('generating SLM model')
    # -- loading PLOS09-type SLM parameter ranges
    fin = open(PLOS09, 'r')
    gen = genson.loads(fin.read())
    fin.close()

    # -- extract SLM parameter range description
    desc = gen.next()
    genson.default_random_seed = DEFAULT_GENSON_SEED

    in_shape = trn_val_ll[0][0][0].shape

    # -- create SLM model
    slm = SequentialLayeredModel(in_shape, desc)

    slm_depth = slm.description[-1][0][1]['initialize']['n_filters']

    # ------------------------------
    # -- Classifier training/testing
    # ------------------------------

    predictions = []
    for idx, trn_val_l in enumerate(trn_val_ll):

        log.info('fold %i of %i' % (idx + 1, len(trn_val_ll)))
        feature_vector_dimension = slm_depth * NBH * NBW

        # -- initialize Classifier
        clf = Classifier(feature_vector_dimension)
        scaler = OnlineScaler(feature_vector_dimension)

        # -- extract lists of training images, training ground truths
        #    validation images and validation ground truths in order
        trn_imgs, trn_gt_imgs, val_imgs, val_gt_imgs = trn_val_l

        # -- train the classifier
        log.info(' training')
        for epoch in xrange(NEPOCH):

            log.info('  epoch %i or %i' % (epoch + 1, NEPOCH))
            trn_list = zip(trn_imgs, trn_gt_imgs)

            if RANDOMIZE:
                shuffle(trn_list)

            for img, gt_img in trn_list:

                # -- training vectors
                f_arr = slm.process(img)
                h_old, w_old = f_arr.shape[:2]
                rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
                h_new, w_new = rf_arr.shape[:2]
                X_trn = rf_arr.reshape(h_new * w_new, -1)
                X_trn = scaler.fit_transform(X_trn)

                # -- labels
                gt_px_x, gt_px_y = slm.rcp_field_central_px_coords(slm.n_layers)
                labels_trn = gt_img[gt_px_x, gt_px_y].reshape(h_old, w_old)
                labels_trn = get_reduced_tm(labels_trn, NBH, NBW).ravel()

                assert X_trn.shape[0] == labels_trn.size

                if RANDOMIZE:
                    idx = np.arange(labels_trn.size)
                    nidx = permutation(idx)
                    X_trn = X_trn[nidx]
                    labels_trn = labels_trn[nidx]

                clf.partial_fit(X_trn, labels_trn)

        # -- validation
        log.info(' validation')
        val_pred, val_gt = [], []
        for img, gt_img in zip(val_imgs, val_gt_imgs):

            # -- training vectors
            f_arr = slm.process(img)
            h_old, w_old = f_arr.shape[:2]
            rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
            h_new, w_new = rf_arr.shape[:2]
            X_val = rf_arr.reshape(h_new * w_new, -1)
            X_val = scaler.fit_transform(X_val)

            # -- True labels
            gt_px_x, gt_px_y = slm.rcp_field_central_px_coords(slm.n_layers)
            labels_val = gt_img[gt_px_x, gt_px_y].reshape(h_old, w_old)
            labels_val = get_reduced_tm(labels_val, NBH, NBW).ravel()

            assert X_val.shape[0] == labels_val.size

            # -- Predicted labels
            labels_val_pred = clf.decision_function(X_val)

            # -- saving predictions and ground truths
            val_pred += [labels_val_pred]
            val_gt += [labels_val]

        # -- test
        log.info(' test')
        tst_imgs, tst_gt_imgs = tst_l
        tst_pred, tst_gt = [], []
        for img, gt_img in zip(tst_imgs, tst_gt_imgs):

            # -- training vectors
            f_arr = slm.process(img)
            h_old, w_old = f_arr.shape[:2]
            rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
            h_new, w_new = rf_arr.shape[:2]
            X_tst = rf_arr.reshape(h_new * w_new, -1)
            X_tst = scaler.fit_transform(X_tst)

            # -- True labels
            gt_px_x, gt_px_y = slm.rcp_field_central_px_coords(slm.n_layers)
            labels_tst = gt_img[gt_px_x, gt_px_y].reshape(h_old, w_old)
            labels_tst = get_reduced_tm(labels_tst, NBH, NBW).ravel()

            assert X_tst.shape[0] == labels_tst.size

            # -- Predicted labels
            labels_tst_pred = clf.decision_function(X_tst)

            # -- saving predictions and ground truths
            tst_pred += [labels_tst_pred]
            tst_gt += [labels_tst]

        # -- storing the predictions for that fold of the cross validation
        predictions.append([val_pred, val_gt, tst_pred, tst_gt])

    # --------------------
    # -- Compute metric(s)
    # --------------------

    log.info('computing metric(s)')
    val_aps = []
    tst_aps = []
    for fold in predictions:

        val_pred, val_gt, tst_pred, tst_gt = fold

        # -- computes metric(s) for validation set
        preds = np.concatenate(val_pred)
        gts = np.concatenate(val_gt)
        val_ap, _ = ap(preds, gts, eps=0.001, preprocess=False)
        val_aps += [val_ap]
        log.info(' validation AP : %4.2f' % val_ap)

        # -- computes metric(s) for testing set
        preds = np.concatenate(tst_pred)
        gts = np.concatenate(tst_gt)
        tst_ap, _ = ap(preds, gts, eps=0.001, preprocess=False)
        tst_aps += [tst_ap]
        log.info(' test AP       : %4.2f' % tst_ap)

    post = {"slm_description": slm.description,
            "val_ap": val_aps,
            "tst_ap": tst_aps}

    posts.insert(post)
