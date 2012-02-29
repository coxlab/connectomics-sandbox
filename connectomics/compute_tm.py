#!/usr/bin/env python

# -- basic imports
import numpy as np
from glob import glob
from os import path
from os import remove
import cPickle

# -- for random utilities
from random import shuffle

# -- utilities
from util import normalize_feature_map
from util import get_trn_tst_coords
from util import split_for_memory
from util import get_reduced_tm
from util import zero_mean_unit_variance
from skimage.shape import view_as_windows
from sklearn.cross_validation import KFold
from skimage.io import imread

# -- for logging
import logging as log
log.basicConfig(level=log.INFO)

# -- for resampling
from skimage.shape import resample

# -- for classifiers
from asgd import NaiveBinaryASGD as Classifier

# -- dataset-related and basic parameters
from coxlabdata.connectome import ConnectomicsHP as Connectome
from parameters import FEATURES_DIR
from parameters import DATASET_PATH
from parameters import IM_SIZE
from parameters import RESIZE_SHAPE

# -- argparse-related
import argparse


def main():
    """front-end with command-line options"""

    parser = argparse.ArgumentParser(description='train classifier'
            ' and predict target maps')
    parser.add_argument(dest='scales', nargs='+', metavar='scales',
            choices=['1024', '512', '256', '128', '64'],
            type=str,
            help='V1 scales to use')
    parser.add_argument(dest='window', nargs=2,
            type=int,
            help='rolling window shape (height, width)')
    parser.add_argument(dest='crossval', nargs=2,
            type=int,
            help='cross validation (nsplit, nfolds)')
    parser.add_argument('--mem_limit', dest='memory_limit', type=int,
            default=1024**3,
            help='limit when unrolling features (default: %(default)s)')
    parser.add_argument('--no_randomize', dest='randomize',
            action='store_false',
            help='disables the randomization of training vectors')
    parser.add_argument('--no_pixels', dest='use_pixels',
            action='store_false',
            help='disables the use of image pixels as extra features')
    parser.add_argument('--epoch', dest='epoch', type=int,
            default=1,
            help='number of pass over training dataset (default: %(default)s)')

    args = vars(parser.parse_args())

    #-------------
    # Main program
    #-------------

    tms = program(args['scales'],
                  args['use_pixels'],
                  args['window'][0], args['window'][1],
                  args['crossval'][0], args['crossval'][1],
                  args['memory_limit'],
                  args['randomize'],
                  args['epoch'])

    #--------------------------------------------
    # Saving the predicted target maps in Pickles
    #--------------------------------------------

    # -- basename for the Pickle files storing the predicted target maps
    scale_string = ''
    for scale in args['scales']:
        scale_string += '_' + scale
    pkl_basename = './asgd_' + str(args['crossval'][1]) + '-fold' + \
                   scale_string + '_window_' + str(args['window'][0]) + \
                   '_' + str(args['window'][1])
    if args['randomize']:
        pkl_basename += '_random'
    if args['use_pixels']:
        pkl_basename += '_withpxs'
    else:
        pkl_basename += '_nopxs'

    for epoch in xrange(args['epoch']):

        log.info('dumping predicted target maps for epoch %i' % (epoch + 1))

        ptm_filename = pkl_basename + '_epoch_%i.pkl' % (epoch + 1)

        if not path.exists(ptm_filename):
            cPickle.dump(tms[epoch], open(ptm_filename, 'wb'))

        else:
            remove(ptm_filename)
            cPickle.dump(tms[epoch], open(ptm_filename, 'wb'))

    return


def program(scales,
            use_pixels,
            h_size, w_size,
            cv_size, cv_nfolds,
            memory_limit,
            randomize,
            nepoch):

    im_size = IM_SIZE
    resample_shape = RESIZE_SHAPE

    #------------
    # some checks
    #------------

    assert cv_size > 1
    assert cv_nfolds <= cv_size
    assert h_size % 2 != 0
    assert w_size % 2 != 0

    #----------------------------
    # Loading feature/target maps
    #----------------------------

    # -- getting the target maps (target maps are binary maps with
    #    values -1 and +1)
    log.info('loading binary target maps')
    dataset_obj = Connectome(DATASET_PATH)
    meta = dataset_obj.meta()
    annotated_meta = sorted([imgd for imgd in meta if 'annotation' in imgd])
    tms = []
    for imgd in annotated_meta:
        tms += [dataset_obj.get_annotation(annd) for annd in imgd['annotation']]

    # -- getting the feature maps
    log.info('loading feature maps')
    farrs = []
    image_basenames = [imgd['annotation'][0]['basename']
                       for imgd in annotated_meta]
    for img_idx, image_basename in enumerate(image_basenames):

        # -- get the list of feature map filenames corresponding to
        #    that image, and corresponding to the chosen scales
        pattern = path.join(FEATURES_DIR, image_basename + '*.npz')
        all_features_fnames = sorted(glob(pattern))
        features_fnames = []
        for fname in all_features_fnames:
            for scale in scales:
                if scale in fname:
                    features_fnames += [fname]
                    break

        # -- resampling the chosen feature arrays
        to_stack = []
        for features_fname in features_fnames:
            log.info('resampling %s' % features_fname)
            arr_in = np.load(features_fname)['features']
            to_stack += [resample(arr_in,
                                  resample_shape,
                                  intp2d=True,
                                  order=0)]

        # -- possibly add the zero mean unit variance original image
        #    to the stack (for extra features)
        if use_pixels:
            grey_scale_img = imread(annotated_meta[img_idx]['filename'],
                                    as_grey=True).astype(np.float32)
            norm_grey_scale_img = zero_mean_unit_variance(grey_scale_img)
            to_stack += [norm_grey_scale_img]

        # -- concatenate into a unique 3D feature array (for the given
        #    image)
        farrs += [np.dstack(tuple(to_stack))]

    assert len(farrs) == len(tms)

    #---------------------------------------
    # initializing the predicted target maps
    #---------------------------------------

    predicted_tms_per_epoch = []
    for epoch in xrange(nepoch):
        predicted_tms = []
        for i in xrange(len(farrs)):
            predicted_tms += [np.empty((im_size - h_size + 1,
                                        im_size - w_size + 1),
                                        dtype=np.float32)]
        predicted_tms_per_epoch.append(predicted_tms)

    #------------------------
    # K-fold cross validation
    #------------------------

    # creating a KFold cross validation iterator
    log.info('cross validation')
    kfcv = KFold(n=cv_size, k=cv_nfolds, indices=False)

    # total number of expected features
    d = farrs[0].shape[-1] * h_size * w_size

    nfolds = 0
    for trn_idx, tst_idx in kfcv:

        #--------------------------
        # classifier initialization
        #--------------------------
        nfolds += 1
        log.info(' fold %3i of %3i' % (nfolds, cv_nfolds))

        clf = Classifier(d)

        # outer loop over 'epoch' number
        for epoch in xrange(nepoch):

            log.info(' epoch %i of %i' % (epoch + 1, nepoch))

            #--------------------
            # classifier training
            #--------------------

            n_pfit = 0
            Xy_list = zip(farrs, tms)
            if randomize:
                shuffle(Xy_list)
            for farr, tm in Xy_list:

                log.info('  partial fit %3i of %3i' % (n_pfit + 1, len(farrs)))

                # -- normalizing the feature map
                norm_farr = normalize_feature_map(farr)

                # -- creating a rolling view on the feature map
                rview = view_as_windows(norm_farr,
                                       (h_size, w_size, norm_farr.shape[-1]))

                # -- reduced target map (target map without the aprons)
                new_tm = get_reduced_tm(tm, h_size, w_size)

                # -- getting the (h, w) coordinates of the training examples
                trn_h, trn_w, _, _ = get_trn_tst_coords(rview.shape[0],
                                                        rview.shape[1],
                                                        trn_idx, tst_idx,
                                                        randomize=randomize)

                # -- training the classifier with mini-batches
                trn_h_l, trn_w_l = split_for_memory(trn_h, trn_w,
                                                    d, rview.itemsize,
                                                    memory_limit)

                for h_arr, w_arr in zip(trn_h_l, trn_w_l):
                    X_train = rview[h_arr, w_arr, :].reshape(h_arr.size, -1)
                    y_train = new_tm[h_arr, w_arr]
                    clf.partial_fit(X_train, y_train)

                # -- increment n_pfit
                n_pfit += 1

            #----------------------
            # classifier prediction
            #----------------------

            n_img = 0
            for farr in farrs:

                log.info('  prediction %3i of %3i' % (n_img + 1, len(farrs)))

                # -- normalizing the feature map
                norm_farr = normalize_feature_map(farr)

                # -- creating a rolling view on the feature map
                rview = view_as_windows(norm_farr,
                                       (h_size, w_size, norm_farr.shape[-1]))

                # -- getting the (h, w) coordinates of the testing examples
                _, _, tst_h, tst_w = get_trn_tst_coords(rview.shape[0],
                                                        rview.shape[1],
                                                        trn_idx, tst_idx,
                                                        randomize=randomize)

                # -- predictions of the classifier in mini-batches
                tst_h_l, tst_w_l = split_for_memory(tst_h, tst_w,
                                                    d, rview.itemsize,
                                                    memory_limit)

                # -- prediction of the classifier
                for h_arr, w_arr in zip(tst_h_l, tst_w_l):
                    X_test = rview[h_arr, w_arr, :].reshape(h_arr.size, -1)
                    y_test = clf.decision_function(X_test)
                    predicted_tms_per_epoch[epoch][n_img][h_arr, w_arr] = y_test

                # -- increment image counter
                n_img += 1

    return predicted_tms_per_epoch


if __name__ == '__main__':
    main()
