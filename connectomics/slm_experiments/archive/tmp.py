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
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np
from numpy.random import permutation
from trn_tst_val_generator import generate
from sthor.model.slm import SequentialLayeredModel
from skimage.util.shape import view_as_windows
import genson
from asgd import NaiveBinaryASGD as Classifier
from scaler import OnlineScaler
from util import get_reduced_tm
from bangmetric import average_precision
from random import shuffle

#from pymongo import Connection
#connection = Connection("localhost:28000")
#db = connection['connectome']
#posts = db.posts

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

from os import path
my_path = path.dirname(path.abspath(__file__))

log.info('generating cross validation sets')
sets_dictionnary = generate(divide_factor=2,
                            trn_val_img_z_idx=DEFAULT_TRN_VAL_IMG_Z_IDX,
                            tst_img_z_idx=DEFAULT_TST_IMG_Z_IDX)

trn_val_ll = sets_dictionnary['trn_val']
tst_l = sets_dictionnary['tst']

# ------------------------------
# -- generation of the SLM model
# ------------------------------

#while True:
if True:

    log.info('generating SLM model')
    # -- loading PLOS09-type SLM parameter ranges
    #gson = path.join(my_path, 'plos09_l3_stride_one.gson')
    #gson = 'test.gson'
    #genson.default_random_seed = DEFAULT_GENSON_SEED
    #with open(gson, 'r') as fin:
        #gen = genson.loads(fin.read())

    # -- extract SLM parameter range description
    #desc = gen.next()
    #from pprint import pprint
    #pprint(desc)

    #outs = genson.dumps(desc, True)
    #open('test.gson', 'w+').write(outs)
    # -- L3 1st
    desc = [
        [('lnorm',
          {'kwargs': {'inker_shape': [9, 9],
                      'outker_shape': [9, 9],
                      'remove_mean': False,
                      'stretch': 10,
                      'threshold': 1}})],
        [('fbcorr',
          #{'initialize': ['426b269c1bfeec366992218fb6e0cb5252cd7f69',
                          #(64, 3, 3)],
          {'initialize': {'filter_shape': (3, 3),
                          'generate': ('random:uniform', {'rseed': 42}),
                          'n_filters': 64},
           'kwargs': {'max_out': None, 'min_out': 0}}),
         ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 1, 'stride': 2}}),
         ('lnorm',
          {'kwargs': {'inker_shape': [5, 5],
                      'outker_shape': [5, 5],
                      'remove_mean': False,
                      'stretch': 0.10000000000000001,
                      'threshold': 1}})],
        [('fbcorr',
          #{'initialize': ['9f1a2ad385682d076a7feacd923e50c330df4e29',
                          #(128, 5, 5, 64)],
          {'initialize': {'filter_shape': (5, 5),
                          'generate': ('random:uniform', {'rseed': 42}),
                          'n_filters': 128},
           'kwargs': {'max_out': None, 'min_out': 0}}),
         ('lpool', {'kwargs': {'ker_shape': [5, 5], 'order': 1, 'stride': 2}}),
         ('lnorm',
          {'kwargs': {'inker_shape': [7, 7],
                      'outker_shape': [7, 7],
                      'remove_mean': False,
                      'stretch': 1,
                      'threshold': 1}})],
        [('fbcorr',
          #{'initialize': ['d79b5af0732b177b2a9170288ba8f73727b56354',
                          #(256, 5, 5, 128)],
          {'initialize': {'filter_shape': (5, 5),
                          'generate': ('random:uniform', {'rseed': 42}),
                          'n_filters': 256},
           'kwargs': {'max_out': None, 'min_out': 0}}),
         ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10, 'stride': 2}}),
         ('lnorm',
          {'kwargs': {'inker_shape': [3, 3],
                      'outker_shape': [3, 3],
                      'remove_mean': False,
                      'stretch': 10,
                      'threshold': 1}})]
    ]

# -- max(1)
#INFO:__main__: validation AP : 0.14
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.10
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.14
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.12
#INFO:__main__: test AP       : 0.12

# -- mean(1)
#INFO:__main__:computing metric(s)
#INFO:__main__: validation AP : 0.19
#INFO:__main__: test AP       : 0.13
#INFO:__main__: validation AP : 0.11
#INFO:__main__: test AP       : 0.13
#INFO:__main__: validation AP : 0.18
#INFO:__main__: test AP       : 0.13
#INFO:__main__: validation AP : 0.12
#INFO:__main__: test AP       : 0.13

# -- scaler + mean(1)
#INFO:__main__: validation AP : 0.18
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.11
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.17
#INFO:__main__: test AP       : 0.12
#INFO:__main__: validation AP : 0.12
#INFO:__main__: test AP       : 0.12

# -- scaler + asgd
#INFO:__main__: validation AP : 0.24
#INFO:__main__: test AP       : 0.30
#INFO:__main__: validation AP : 0.31
#INFO:__main__: test AP       : 0.30
#INFO:__main__: validation AP : 0.25
#INFO:__main__: test AP       : 0.31
#INFO:__main__: validation AP : 0.29
#INFO:__main__: test AP       : 0.30

    #raise

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
                #img -= img.min()
                #img /= img.max()
                f_arr = slm.process(img)
                h_old, w_old = f_arr.shape[:2]
                rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
                h_new, w_new = rf_arr.shape[:2]
                X_trn = rf_arr.reshape(h_new * w_new, -1)
                X_trn = scaler.partial_fit(X_trn).transform(X_trn)

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
            #img -= img.min()
            #img /= img.max()
            f_arr = slm.process(img)
            h_old, w_old = f_arr.shape[:2]
            rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
            h_new, w_new = rf_arr.shape[:2]
            X_val = rf_arr.reshape(h_new * w_new, -1)
            X_val = scaler.transform(X_val)

            # -- True labels
            gt_px_x, gt_px_y = slm.rcp_field_central_px_coords(slm.n_layers)
            labels_val = gt_img[gt_px_x, gt_px_y].reshape(h_old, w_old)
            labels_val = get_reduced_tm(labels_val, NBH, NBW).ravel()

            assert X_val.shape[0] == labels_val.size

            # -- Predicted labels
            labels_val_pred = clf.decision_function(X_val)
            #labels_val_pred = X_val.max(1)

            # -- saving predictions and ground truths
            val_pred += [labels_val_pred]
            val_gt += [labels_val]

        # -- test
        log.info(' test')
        tst_imgs, tst_gt_imgs = tst_l
        tst_pred, tst_gt = [], []
        for img, gt_img in zip(tst_imgs, tst_gt_imgs):

            # -- training vectors
            #img -= img.min()
            #img /= img.max()
            f_arr = slm.process(img)
            h_old, w_old = f_arr.shape[:2]
            rf_arr = view_as_windows(f_arr, (NBH, NBW, 1))
            h_new, w_new = rf_arr.shape[:2]
            X_tst = rf_arr.reshape(h_new * w_new, -1)
            #X_tst = scaler.fit_transform(X_tst)
            X_tst = scaler.partial_fit(X_tst).transform(X_tst)

            # -- True labels
            gt_px_x, gt_px_y = slm.rcp_field_central_px_coords(slm.n_layers)
            labels_tst = gt_img[gt_px_x, gt_px_y].reshape(h_old, w_old)
            labels_tst = get_reduced_tm(labels_tst, NBH, NBW).ravel()

            assert X_tst.shape[0] == labels_tst.size

            # -- Predicted labels
            labels_tst_pred = clf.decision_function(X_tst)
            #labels_tst_pred = X_tst.max(1)

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
        val_ap = average_precision(gts, preds)
        val_aps += [val_ap]
        log.info(' validation AP : %4.2f' % val_ap)

        # -- computes metric(s) for testing set
        preds = np.concatenate(tst_pred)
        gts = np.concatenate(tst_gt)
        tst_ap = average_precision(gts, preds)
        tst_aps += [tst_ap]
        log.info(' test AP       : %4.2f' % tst_ap)

    post = {"slm_description": slm.description,
            "val_ap": val_aps,
            "tst_ap": tst_aps}
    print post

    #posts.insert(post)
