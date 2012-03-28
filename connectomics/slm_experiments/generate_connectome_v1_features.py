# -- basic imports
import numpy as np
from os import path
import cPickle
import os

# -- connectome dataset object
from coxlabdata.connectome import ConnectomicsHP as Connectome

# -- V1-like related
from v1like import config, v1like_extract
from v1like.v1like_funcs import get_image

# -- resampling utility
from sthor.operation import resample

# -- global parameters for feature extraction
from parameters import DATASET_PATH
from parameters import V1_FEATURES_FILENAME
from parameters import MAX_EDGE_L
from parameters import RESIZE_SHAPE
from parameters import OVERWRITE
from parameters import V1_MODEL_CONFIG

# ----------------------------------------------------------------------------
# -- Main entry
# ----------------------------------------------------------------------------
def main():

    # -- retrieving the meta data concerning the "ConnectomeHP" dataset
    ConnectomeHP_dataset_object = Connectome(DATASET_PATH)
    meta = ConnectomeHP_dataset_object.meta()

    # -- print some information on screen
    print len(meta), 'images to process'
    print 'Storing all V1 features into:', V1_FEATURES_FILENAME

    # -- getting image filenames for later processing
    img_filenames = [imgd['filename'] for imgd in meta]

    # -- getting the scales at which to compute V1 features
    scales = MAX_EDGE_L

    # -- opening the file that will contain the V1 features in
    #    binary format
    if not path.exists(V1_FEATURES_FILENAME):
        v1_file = open(V1_FEATURES_FILENAME, 'wb')
    else:
        if OVERWRITE:
            os.remove(V1_FEATURES_FILENAME)
            v1_file = open(V1_FEATURES_FILENAME, 'wb')
        else:
            return

    # -- processing the images to extract the V1 features
    #    and dumping to file
    for img_fname in img_filenames:

        print 'processing image:', path.basename(img_fname)

        for scale in scales:

            print 'processing for scale:', scale

            # -- compute the 'raw' features
            print '> feature extraction'
            farr = get_V1_features(img_fname, scale, V1_MODEL_CONFIG)

            # -- resample the feature array to the common shape
            print '> resampling to %s' % str(RESIZE_SHAPE)
            farr = resample(farr, RESIZE_SHAPE, order=0, intp2d=True)

            # -- dumping the feature array to file in binary mode
            farr.tofile(v1_file)

    v1_file.close()

    # -- dumping a Pickle file that will contain a dictionnary
    #    explaining how the above-generated file should be
    #    memory mapped with numpy
    dico = {'image_number': len(img_filenames),
            'scale_number': len(scales),
            'scale_sizes': scales,
            '(h, w, d)': RESIZE_SHAPE,
            'global_memmap_array_shape': (len(img_filenames),
                                          len(scales)) +
                                          RESIZE_SHAPE,
            'dtype': farr.dtype}

    cPickle.dump(dico, open(V1_FEATURES_FILENAME + '_dict.pkl', 'w'))

    return

# ----------------------------------------------------------------------------
# -- Helpers
# ----------------------------------------------------------------------------

def get_V1_features(img_fname, scale, v1like_config,
                    with_pool_outshape=False,
                    with_pool_lsum_ksize=False):

    # -- V1 configuration to use
    rep, featsel = config.get(v1like_config, verbose=False)

    # -- get image to right scale
    resize_method = rep['preproc']['resize_method']
    arr = get_image(img_fname, max_edge=scale,
                    resize_method=resize_method)
    arr = np.atleast_3d(arr).mean(2)

    # -- get rid of some resize-related preproc parameters
    del rep['preproc']['max_edge']
    del rep['preproc']['resize_method']

    if not with_pool_outshape:
        del rep['pool']['outshape']
    if not with_pool_lsum_ksize:
        del rep['pool']['lsum_ksize']

    # -- actually compute the features
    farr = v1like_extract.v1like_fromarray(arr, rep, featsel,
                                           ravel_it=False,
                                           use_fft_cache=True)

    return farr


if __name__ == '__main__':
    main()
