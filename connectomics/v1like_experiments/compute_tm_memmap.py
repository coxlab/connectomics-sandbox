#!/usr/bin/env python

# -- for logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -- basic imports
import numpy as np
from os import path
from os import remove
import cPickle

# -- for random utilities
from random import shuffle

# -- utilities
from util import get_reduced_tm
from util import get_memmap_array
from util import generate_Xy_train
from util import generate_Xhw_test
from util import OnlineScaler
from sklearn.cross_validation import KFold

# -- for classifiers
from asgd import NaiveBinaryASGD as Classifier

# -- dataset-related and basic parameters
from coxlabdata import ConnectomicsHP
from parameters import V1_FEATURES_FILENAME
from parameters import IM_SIZE

# -- argparse-related
import argparse


def main():
    """front-end with command-line options"""

    parser = argparse.ArgumentParser(description='train classifier'
            ' and predict target maps')
    parser.add_argument(dest='scales', nargs=1,
            choices=[1, 2, 3, 4, 5],
            type=int,
            help='number of scales to use (1-5)')
    parser.add_argument(dest='window', nargs=3,
            type=int,
            help='rolling window shape (z, height, width)')
    parser.add_argument(dest='crossval', nargs=2,
            type=int,
            help='cross validation (nsplit, nfolds)')
    parser.add_argument('--mem_limit', dest='memory_limit', type=int,
            default=2*1024**3,
            help='limit when unrolling features (default: %(default)s)')
    parser.add_argument('--no_randomize', dest='randomize',
            action='store_false',
            help='disables the randomization of training vectors')
    parser.add_argument('--epoch', dest='epoch', type=int,
            default=1,
            help='number of pass over training dataset (default: %(default)s)')

    args = vars(parser.parse_args())

    #-------------
    # Main program
    #-------------

    tms = program(args['scales'][0],
                  args['window'][0], args['window'][1], args['window'][2],
                  args['crossval'][0], args['crossval'][1],
                  args['memory_limit'],
                  args['randomize'],
                  args['epoch'])

    #--------------------------------------------
    # Saving the predicted target maps in Pickles
    #--------------------------------------------

    # -- basename for the Pickle files storing the predicted target maps
    scale_list = ['1024', '512', '256', '128', '64']
    scale_string = '_scales'
    for scale in range(args['scales'][0]):
        scale_string += '_' + scale_list[scale]
    pkl_basename = './asgd_' + str(args['crossval'][1]) + '-fold' + \
                   scale_string + '_window_' + str(args['window'][0]) + \
                   '_' + str(args['window'][1]) + '_' + str(args['window'][2])
    if args['randomize']:
        pkl_basename += '_random'

    for epoch in xrange(args['epoch']):

        log.info('dumping predicted target maps for epoch %i' % (epoch + 1))

        ptm_filename = pkl_basename + '_epoch_%i.pkl' % (epoch + 1)

        if not path.exists(ptm_filename):
            cPickle.dump(tms[epoch], open(ptm_filename, 'wb'))

        else:
            remove(ptm_filename)
            cPickle.dump(tms[epoch], open(ptm_filename, 'wb'))

    return


def program(n_scales,
            z_size, h_size, w_size,
            cv_size, cv_nfolds,
            memory_limit,
            randomize,
            nepoch):

    im_size = IM_SIZE

    #------------
    # some checks
    #------------

    assert cv_size > 1
    assert n_scales >= 1
    assert cv_nfolds <= cv_size
    assert z_size % 2 != 0
    assert h_size % 2 != 0
    assert w_size % 2 != 0

    #----------------------------
    # Loading feature/target maps
    #----------------------------

    # -- getting the target maps (target maps are binary maps with
    #    values -1 and +1)
    log.info('loading binary target maps')
    dataset_obj = ConnectomicsHP()
    meta = dataset_obj.meta()

    annotated_meta = sorted([imgd for imgd in meta if 'annotation' in imgd])

    tms, trn_img_idx_rng = [], []
    for imgd in annotated_meta:
        binary_tm = [dataset_obj.get_annotation(annd)
                     for annd in imgd['annotation']]
        tms += [np.where(btm == False, -1., 1.).astype(np.float32)
                     for btm in binary_tm]
        trn_img_idx_rng += [(imgd['z'] - 1 - (z_size / 2),
                             imgd['z'] - 1 + (z_size / 2) + 1)]

    # -- number of training images
    n_trn_imgs = len(tms)

    # -- getting access to the whole dataset features. The shape of this
    #    array is (#images, #scales, h, w, d)
    log.info('building memmap feature array')
    arr = get_memmap_array(V1_FEATURES_FILENAME)

    #---------------------------------------
    # initializing the predicted target maps
    #---------------------------------------

    log.info('initializing final target maps')
    predicted_tms_per_epoch = []
    for epoch in xrange(nepoch):
        predicted_tms = []
        for i in xrange(n_trn_imgs):
            predicted_tms += [np.empty((im_size - h_size + 1,
                                        im_size - w_size + 1),
                                        dtype=np.float32)]
        predicted_tms_per_epoch.append(predicted_tms)

    #------------------------
    # K-fold cross validation
    #------------------------

    # creating a KFold cross validation iterator
    log.info('building the cross validation')
    kfcv = KFold(n=cv_size, k=cv_nfolds, indices=False)

    # total number of expected features
    d = n_scales * z_size * h_size * w_size * arr.shape[-1]
    log.info('feature dimensionality: %i' % d)

    nfolds = 0
    for trn_idx, tst_idx in kfcv:

        #--------------------------
        # classifier initialization
        #--------------------------
        nfolds += 1
        log.info(' fold %3i of %3i' % (nfolds, cv_nfolds))

        clf = Classifier(d)
        scaler = OnlineScaler(d)

        # outer loop over 'epoch' number
        for epoch in xrange(nepoch):

            log.info(' epoch %i of %i' % (epoch + 1, nepoch))

            #--------------------
            # classifier training
            #--------------------

            n_pfit = 0
            Xy_list = zip(trn_img_idx_rng, tms)
            if randomize:
                shuffle(Xy_list)
            for (img_idx_start, img_idx_stop), tm in Xy_list:

                log.info('  partial fit %3i of %3i' % (n_pfit + 1, n_trn_imgs))

                # -- reduced target map (target map without the aprons)
                new_tm = get_reduced_tm(tm, h_size, w_size)

                # -- extracting the appropriate feature array(s)
                farr = arr[img_idx_start:img_idx_stop, :n_scales]

                # -- memory-managed generator for (X, y) training examples
                Xy_trn_l = generate_Xy_train(farr,
                                             h_size, w_size,
                                             trn_idx, tst_idx,
                                             new_tm,
                                             memory_limit,
                                             randomize)

                for X_train, y_train in Xy_trn_l:
                    X_train = scaler.fit_transform(X_train)
                    clf.partial_fit(X_train, y_train)

                # -- increment n_pfit
                n_pfit += 1

            #----------------------
            # classifier prediction
            #----------------------

            n_img = 0
            for (img_idx_start, img_idx_stop) in trn_img_idx_rng:

                log.info('  prediction %3i of %3i' % (n_img + 1, n_trn_imgs))

                # -- extracting the appropriate feature array
                farr = arr[img_idx_start:img_idx_stop, :n_scales]

                # -- memory-managed generator for X testing examples
                #    it also generates some (h, w) coordinate arrays
                #    so as to easily reinsert the predicted y_test
                #    into predicted target map arrays
                Xhw_tst_l = generate_Xhw_test(farr,
                                              h_size, w_size,
                                              trn_idx, tst_idx,
                                              memory_limit,
                                              randomize)

                # -- prediction of the classifier
                for X_test, h_arr, w_arr in Xhw_tst_l:
                    X_test = scaler.transform(X_test)
                    y_test = clf.decision_function(X_test)
                    predicted_tms_per_epoch[epoch][n_img][h_arr, w_arr] = y_test

                # -- increment image counter
                n_img += 1

    return predicted_tms_per_epoch


if __name__ == '__main__':
    main()
