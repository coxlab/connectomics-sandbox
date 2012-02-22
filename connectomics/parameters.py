#!/usr/bin/env python

"""
Gathers all the parameters used for model selection and more
"""

# full path to where the connectome dataset is on disk
DATASET_PATH = '/share/datasets/connectomics2011'

# where to save the *.npz files containing the features
FEATURES_DIR = '/home/npoilvert/connectome/v1like_features'

# name of the file containing all the v1 like feature of
# all images in the dataset for all scales (given by the
# max edge length)
V1_FEATURES_FILENAME = '/share/users/npoilvert/' + \
                       'connectome_V1_5_scale_features.dat'

# imsize
IM_SIZE = 1024

# common shape of feature maps
RESIZE_SHAPE = (1024, 1024, 96)

# sizes to which images will be rescaled
MAX_EDGE_L = [1024, 1024 / 2, 1024 / 4,
              1024 / 8, 1024 / 16]

# overwrite features if recomputed
OVERWRITE = False

# v1like type of model
V1_MODEL_CONFIG = 'v1like_a'
