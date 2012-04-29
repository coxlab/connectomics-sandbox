#!/usr/bin/env python

"""
This program will:
    1. create a cross validated screening task with
       train/validation and testing sets
    2. trains a classifier for every cross validation
       fold and computes the performance for both the
       validation images and testing images
    3. computes the metrics

This code uses the latest SLM model that can compute
feature vectors for all pixels.

It also uses many classifiers to help separate membrane
directions in different "channels"
"""

# -- for logging
import logging as log
log.basicConfig(level=log.INFO)

import cPickle
from random import choice
import numpy as np
from numpy.random import permutation
from trn_tst_val_generator import generate
from sthor.model.slm import SequentialLayeredModel
import genson
from asgd import NaiveBinaryASGD as Classifier
from scaler import OnlineScaler
from util import get_trn_coords_labels
from util import predict
from bangmetric.precision_recall import average_precision
from bangmetric.correlation import pearson, spearman
from random import shuffle

from pymongo import Connection

connection = Connection('localhost',28000)
db = connection['connectome']
coll = db['slm_models']

# -----------
# -- defaults
# -----------

PLOS09 = './plos09.gson'
DEFAULT_FB_FNAME = '9_9_2.pkl'
DEFAULT_TRN_VAL_IMG_Z_IDX = [71]
DEFAULT_TST_IMG_Z_IDX = [3]
DEFAULT_GENSON_SEED = 5
USE_VERENA = True
RANDOMIZE = True
NEPOCH = 1
NEGPOSFRAC = 2.
USE_SUPERCLF = choice([True, False])

# -------------------------------------------------------
# Loading the filter bank obtained with KMeans clustering
# -------------------------------------------------------

log.info('loading KMeans filter bank')
with open(DEFAULT_FB_FNAME, 'r') as fname:
    fb = cPickle.load(fname)
fbh, fbw, nclf = fb.shape
log.info('%i filters of shape (%i, %i)' % (nclf, fbh, fbw))

# -------------------------------------------------
# -- generate training, validation and testing sets
# -------------------------------------------------

log.info('generating cross validation sets')
sets_dictionnary = generate(divide_factor=2,
                            trn_val_img_z_idx=DEFAULT_TRN_VAL_IMG_Z_IDX,
                            tst_img_z_idx=DEFAULT_TST_IMG_Z_IDX,
                            use_verena=USE_VERENA)

trn_val_ll = sets_dictionnary['trn_val']
tst_l = sets_dictionnary['tst']

if USE_VERENA:
    verena_imgs = sets_dictionnary['verena_l']

# ------------------------------
# -- generation of the SLM model
# ------------------------------

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

feature_vector_dimension = slm.description[-1][0][1]['initialize']['n_filters']

# ------------------------------
# -- Classifier training/testing
# ------------------------------

predictions = []
for idx, trn_val_l in enumerate(trn_val_ll):

    log.info('fold %i of %i' % (idx + 1, len(trn_val_ll)))

    # -- initialize Classifier bank
    clf_b = []
    for _ in xrange(nclf):
        clf_b += [Classifier(feature_vector_dimension)]

    # -- initialize Scaler
    scaler = OnlineScaler(feature_vector_dimension)

    # -- extract lists of training images, training ground truths
    #    validation images and validation ground truths in order
    trn_imgs, trn_gt_imgs, val_imgs, val_gt_imgs = trn_val_l

    # -- train the bank of classifiers
    log.info(' training')
    for epoch in xrange(NEPOCH):

        log.info('  epoch %i of %i' % (epoch + 1, NEPOCH))
        trn_list = zip(trn_imgs, trn_gt_imgs)

        if RANDOMIZE:
            shuffle(trn_list)

        if USE_SUPERCLF:
            X_trn_y_trn_l = []

        for img, gt_img in trn_list:

            # -- training vectors
            f_arr = slm.process(img, pad_apron=True, interleave_stride=True)
            X_trn = f_arr.reshape((f_arr.shape[0] * f_arr.shape[1],
                                   f_arr.shape[2]))
            X_trn = scaler.fit_transform(X_trn)

            # -- training labels
            y_trn = gt_img.ravel()

            assert X_trn.shape[0] == y_trn.size
            assert X_trn.shape[1] == feature_vector_dimension

            if RANDOMIZE:
                idx = permutation(np.arange(y_trn.size))
                X_trn = X_trn[idx]
                y_trn = y_trn[idx]

            # -- save X_trn and y_trn for 'super-classifier' training
            if USE_SUPERCLF:
                X_trn_y_trn_l += [(X_trn.copy(), y_trn.copy())]

            # -- compute multi-classifier training examples
            coords_labels_l = get_trn_coords_labels(gt_img, fb,
                                                    fraction=NEGPOSFRAC)

            # -- loop over classifiers (here it is important to note that
            #    f_arr has been properly scaled because X_trn was a *view*
            #    on f_arr and it was itself scaled)
            for i, (x_coords, y_coords, labels) in enumerate(coords_labels_l):
                X = f_arr[x_coords, y_coords]
                clf_b[i].partial_fit(X, labels)

    # -- train 'super classifier' if need be
    if USE_SUPERCLF:
        log.info('  super classifier training')
        s_clf = Classifier(nclf)
        s_scaler = OnlineScaler(nclf)
        for X_trn, y_trn in X_trn_y_trn_l:
            X_new = predict(X_trn, clf_b)
            X_new = s_scaler.fit_transform(X_new)
            s_clf.partial_fit(X_new, y_trn)

    # -- validation
    log.info(' validation')
    val_pred, val_gt = [], []

    for img, gt_img in zip(val_imgs, val_gt_imgs):

        # -- training vectors
        f_arr = slm.process(img, pad_apron=True, interleave_stride=True)
        X_val = f_arr.reshape((f_arr.shape[0] * f_arr.shape[1],
                               f_arr.shape[2]))
        X_val = scaler.transform(X_val)

        # -- True labels
        y_val = gt_img.ravel()

        assert X_val.shape[0] == y_val.size
        assert X_val.shape[1] == feature_vector_dimension

        # -- Predicted labels
        if USE_SUPERCLF:
            X_new = predict(X_val, clf_b)
            X_new = s_scaler.transform(X_new)
            y_val_pred = s_clf.decision_function(X_new)
        else:
            X_new = predict(X_val, clf_b)
            y_val_pred = X_new.max(axis=1)

        # -- saving predictions and ground truths
        assert y_val.size == y_val_pred.size
        val_pred += [y_val_pred]
        val_gt += [y_val]

    # -- test
    log.info(' test')
    tst_imgs, tst_gt_imgs = tst_l
    tst_pred, tst_gt = [], []

    if USE_VERENA:
        verena_pred = []
        counter = 0

    for img, gt_img in zip(tst_imgs, tst_gt_imgs):

        # -- training vectors
        f_arr = slm.process(img, pad_apron=True, interleave_stride=True)
        X_tst = f_arr.reshape((f_arr.shape[0] * f_arr.shape[1],
                               f_arr.shape[2]))
        X_tst = scaler.transform(X_tst)

        # -- True labels
        y_tst = gt_img.ravel()

        if USE_VERENA:
            verena_tst = verena_imgs[counter]
            verena_tst = verena_tst.ravel()
            counter += 1

        assert X_tst.shape[0] == y_tst.size
        assert X_tst.shape[1] == feature_vector_dimension

        # -- Predicted labels
        if USE_SUPERCLF:
            X_new = predict(X_tst, clf_b)
            X_new = s_scaler.transform(X_new)
            y_tst_pred = s_clf.decision_function(X_new)
        else:
            X_new = predict(X_tst, clf_b)
            y_tst_pred = X_new.max(axis=1)

        # -- saving predictions and ground truths
        assert y_tst.size == y_tst_pred.size
        tst_pred += [y_tst_pred]
        tst_gt += [y_tst]

        if USE_VERENA:
            verena_pred += [verena_tst]

    # -- storing the predictions for that fold of the cross validation
    if USE_VERENA:
        predictions.append([val_pred, val_gt, tst_pred, tst_gt, verena_pred])
    else:
        predictions.append([val_pred, val_gt, tst_pred, tst_gt])

# --------------------
# -- Compute metric(s)
# --------------------

log.info('computing metric(s)')
val_aps, val_pearsons, val_spearmans = [], [], []
tst_aps, tst_pearsons, tst_spearmans = [], [], []

if USE_VERENA:
    verena_aps, verena_pearsons, verena_spearmans = [], [], []

for fold in predictions:

    if USE_VERENA:
        val_pred, val_gt, tst_pred, tst_gt, verena_pred = fold
    else:
        val_pred, val_gt, tst_pred, tst_gt = fold

    # -- computes metric(s) for validation set
    gts, preds = np.concatenate(val_gt), np.concatenate(val_pred)
    val_ap = average_precision(gts, preds)
    val_pearson = pearson(gts, preds)
    val_spearman = spearman(gts, preds)
    val_aps += [val_ap]
    val_pearsons += [val_pearson]
    val_spearmans += [val_spearman]
    log.info(' validation AP       : %5.3f' % val_ap)
    log.info(' validation Pearson  : %5.3f' % val_pearson)
    log.info(' validation Spearman : %5.3f' % val_spearman)

    # -- computes metric(s) for testing set
    gts, preds = np.concatenate(tst_gt), np.concatenate(tst_pred)
    tst_ap = average_precision(gts, preds)
    tst_pearson = pearson(gts, preds)
    tst_spearman = spearman(gts, preds)
    tst_aps += [tst_ap]
    tst_pearsons += [tst_pearson]
    tst_spearmans += [tst_spearman]
    log.info(' test AP       : %5.3f' % tst_ap)
    log.info(' test Pearson  : %5.3f' % tst_pearson)
    log.info(' test Spearman : %5.3f' % tst_spearman)

    if USE_VERENA:
        gts, preds = np.concatenate(tst_gt), np.concatenate(verena_pred)
        verena_ap = average_precision(gts, preds)
        verena_pearson = pearson(gts, preds)
        verena_spearman = spearman(gts, preds)
        verena_aps += [verena_ap]
        verena_pearsons += [verena_pearson]
        verena_spearmans += [verena_spearman]
        log.info(' Verena AP       : %5.3f' % verena_ap)
        log.info(' Verena Pearson  : %5.3f' % verena_pearson)
        log.info(' Verena Spearman : %5.3f' % verena_spearman)

post = {"epoch": NEPOCH,
        "genson_seed": DEFAULT_GENSON_SEED,
        "trn_val_img_idx": DEFAULT_TRN_VAL_IMG_Z_IDX,
        "tst_img_idx": DEFAULT_TST_IMG_Z_IDX,
        "randomize": RANDOMIZE,
        "slm_description": slm.description,
        "val_ap": val_aps,
        "tst_ap": tst_aps,
        "verena_ap": verena_aps,
        "val_pearson": val_pearsons,
        "tst_pearson": tst_pearsons,
        "verena_pearson": verena_pearsons,
        "val_spearman": val_spearmans,
        "tst_spearman": tst_spearmans,
        "verena_spearman": verena_spearmans,
        "use_super-classifier": USE_SUPERCLF,
        "negative_to_positive_example_ratio": NEGPOSFRAC
        }

coll.insert(post)
