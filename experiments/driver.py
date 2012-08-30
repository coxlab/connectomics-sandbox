#!/usr/bin/env python

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>

# Licence: BSD

# -- external modules
from get_isbi_images import get_images as isbi_images
from get_hp_images import get_images as hp_images
from bangmetric import precision_recall
from bangmetric import correlation
from bangmetric.wildwest.isbi12 import pixel_error, rand_error, warp_error

# -- basic imports
import sys
import time
import cPickle
import numpy as np
from os import path
import argparse
from pymongo import Connection

# -- Default parameters
# default name for the Pickle file that will store the data
DEFAULT_PKL_FNAME = 'saved_data.pkl'

# training images to use (should be a list of integers)
DEFAULT_TRN_IDX = range(30)

# testing images to use (should be a list of integers)
DEFAULT_TST_IDX = range(30)

# whether the driver should save data by default ?
DEFAULT_SAVE = False

# whether to add the training images rotated by 90, 180 and 270 degres to the
# training set ?
DEFAULT_ROTATE = False

# if "True", the actual "official" testing images are used instead of validation
# images from the training set (this works only for the ISBI dataset)
DEFAULT_USE_TRUE_TST_IMG = True

# -- Set your MongoDB-related parameters here
# should the driver store in a MongoDB database by default ?
DEFAULT_MONGO_STORE = True

# what port to use to connect to the MongoDB database ?
DEFAULT_MONGO_PORT = 28000

# what is the IP address of the server holding the MongoDB database ?
DEFAULT_MONGO_HOST = "<my_host_IP_address>"

# what is the default MongoDB database name ?
DEFAULT_MONGO_DB = '<my_mongo_database>'

# to which database collection should we push the data to ?
DEFAULT_MONGO_COLL = '<my_collection_name>'

# used for caching the training/testing images
CACHE = {}


def main():
    """
    program front-end with command-line options
    """

    parser = argparse.ArgumentParser(description='driver program for' + \
                                     ' Connectomics')

    # -- positional arguments
    parser.add_argument('function', action='store')
    parser.add_argument('function_arguments', nargs='*',
                        help='args of the function to pass to the driver')

    # -- non-positional arguments
    parser.add_argument('--trn_img_idx', action='append', dest='trn_img_idx',
                        type=int,
                        default=DEFAULT_TRN_IDX,
                        help='z coordinate of training images to consider',
                        )
    parser.add_argument('--tst_img_idx', action='append', dest='tst_img_idx',
                        type=int,
                        default=DEFAULT_TST_IDX,
                        help='z coordinate of testing images to consider',
                        )
    parser.add_argument('--hp_dataset', action='store_true',
                        default=False,
                        dest='hp_dataset',
                        help='flag to select Hans Pfister dataset instead of' \
                             + ' ISBI')
    parser.add_argument('--save', action='store_true',
                        default=DEFAULT_SAVE,
                        dest='save',
                        help='flag to switch on data recording')
    parser.add_argument('--pkl_fname',
                        default=DEFAULT_PKL_FNAME,
                        dest='pkl_fname',
                        help='name of the Pickle file containing the data')
    parser.add_argument('--no_mongo_store', action='store_false',
                        default=DEFAULT_MONGO_STORE,
                        dest='mongo',
                        help='disables storage into MongoDB')

    # -- program version
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()

    # -- building the train/test data (with caching)
    if args.hp_dataset:
        key = (tuple(args.trn_img_idx), tuple(args.tst_img_idx), 'hp')
        if key not in CACHE:
            tasks = hp_images(trn_img_idx=args.trn_img_idx,
                              tst_img_idx=args.tst_img_idx)
            CACHE[key] = tasks
        else:
            tasks = CACHE[key]
    else:
        key = (tuple(args.trn_img_idx), tuple(args.tst_img_idx), 'isbi')
        if key not in CACHE:
            tasks = isbi_images(trn_img_idx=args.trn_img_idx,
                                tst_img_idx=args.tst_img_idx,
                                rotate_img=DEFAULT_ROTATE,
                                use_true_tst_img=DEFAULT_USE_TRUE_TST_IMG)
            CACHE[key] = tasks
        else:
            tasks = CACHE[key]

    # -- call user's processing function
    start = time.time()
    function = import_function(args.function)
    output_true, output_pred, to_save = function(tasks, *args.function_arguments)
    stop = time.time()

    # -- compute metrics from Ground Truth and Predictions
    assert output_pred.ndim == 4
    n_images = output_pred.shape[0]
    n_cv_folds = output_pred.shape[-1]

    if len(output_true) > 0:

        assert output_true.shape == output_pred.shape

        pearsons, aps = [], []
        pxs, rds, wps = [], [], []

        for j in xrange(n_cv_folds):

            cv_pearsons, cv_aps = [], []
            cv_pxs, cv_rds, cv_wps = [], [], []

            for i in xrange(n_images):

                y_true = output_true[i, :, :, j]
                y_pred = output_pred[i, :, :, j]
                cv_pearsons += [correlation.pearson(y_true.ravel(), y_pred.ravel())]
                cv_aps += [precision_recall.average_precision(y_true.ravel(), y_pred.ravel())]
                cv_pxs += [pixel_error(y_true, y_pred, th_min=0.5, th_max=0.6, th_inc=0.1)]
                cv_rds += [rand_error(y_true, y_pred, th_min=0.5, th_max=0.6, th_inc=0.1)]
                cv_wps += [warp_error(y_true, y_pred, th_min=0.5, th_max=0.6, th_inc=0.1)]

            pearsons += [cv_pearsons]
            aps += [cv_aps]
            pxs += [cv_pxs]
            rds += [cv_rds]
            wps += [cv_wps]

        pearsons = np.array(pearsons)
        aps = np.array(aps)
        pxs = np.array(pxs)
        rds = np.array(rds)
        wps = np.array(wps)

        # -- reporting metrics mean values
        print 'mean Average Precision: %6.4f' % aps.mean()
        print 'mean Pearson coef.    : %6.4f' % pearsons.mean()
        print 'mean Pixel Error      : %6.4f' % pxs.mean()
        print 'mean Rand Error       : %6.4f' % rds.mean()
        print 'mean Warping Error    : %6.4f' % wps.mean()

    print 'time to compute (s)   : %6.3f' % (stop - start)

    if args.save:

        if args.mongo:

            connection = Connection(DEFAULT_MONGO_HOST, DEFAULT_MONGO_PORT)
            db = connection[DEFAULT_MONGO_DB]
            coll = db[DEFAULT_MONGO_COLL]

            if len(output_true) > 0:
                mongo_post = dict(
                                  dataset_parameters=args.__dict__,
                                  ap=aps.tolist(),
                                  px=pxs.tolist(),
                                  rd=rds.tolist(),
                                  wp=wps.tolist(),
                                  pearson=pearsons.tolist(),
                                  time=(stop - start),
                                  fct_parameters=to_save,
                                 )
                coll.insert(mongo_post)
            else:
                mongo_post = dict(
                                  dataset_parameters=args.__dict__,
                                  time=(stop - start),
                                  fct_parameters=to_save,
                                 )
                coll.insert(mongo_post)

        else:
            with open(args.pkl_fname, 'w') as f:
                if len(output_true) > 0:
                    pkl_content = dict(
                                       output_true=output_true,
                                       output_pred=output_pred,
                                       dataset_parameters=args.__dict__,
                                       ap=aps,
                                       px=pxs,
                                       rd=rds,
                                       wp=wps,
                                       pearson=pearsons,
                                       time=(stop - start),
                                       fct_parameters=to_save,
                                      )
                    cPickle.dump(pkl_content, f)
                else:
                    pkl_content = dict(
                                       output_pred=output_pred,
                                       dataset_parameters=args.__dict__,
                                       time=(stop - start),
                                       fct_parameters=to_save,
                                      )
                    cPickle.dump(pkl_content, f)

    return


# -- helper function to import the proper function
def import_function(function_path):

    directory, function_path = path.split(function_path)
    sys.path.append(directory)

    assert len(function_path.split('.')) == 2
    module_name, function_name = function_path.split('.')

    module = __import__(module_name)

    function = getattr(module, function_name)

    return function


if __name__ == '__main__':
    main()
