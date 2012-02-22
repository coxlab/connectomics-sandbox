# -*- coding: utf-8 -*-
"""
Connectome Dataset
"""

# Copyright (C) 2011
# Authors: Nicolas Poilvert <poilvert@rowland.harvard.edu>

# License: Simplified BSD

import os
from os import path
import re
import shutil
from glob import glob
import hashlib
import numpy as np
from scipy.misc import imread
from PIL import Image

from skdata.data_home import get_data_home


class ConnectomeBase(object):
    """Connectome Object Dataset

    Attributes
    ----------
    src: string
        gives the path to the dataset on disk

    name: string
        name of the dataset (defaults to the class name)

    meta: list of dict
        metadata associated with the dataset. For each image with index `i`,
        `meta[i]` is a dict with keys:
            filename: str
                full path to the image
            z: int or None
                'z' coordinate of the image in the stack
            sha1: str
                SHA-1 hash of the image
            shape: tuple
                shape of the image
            annotated: bool
                True if the image has annotation, False otherwise
            annotation_filename: str
                if 'annotated' is True, gives the full path to the annotated
                image. If 'annotated' is False, gives None
    """

    def __init__(self, src, meta=None):

        if meta is not None:
            self._meta = meta

        self.src = src
        self.name = self.__class__.__name__
        self.image_dir_basename = 'Images'

    def home(self, *suffix_paths):
        """returns home full path"""
        return path.join(get_data_home(), self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self):
        """moves the full dataset into the skdata default directory XXX: mofidy
        the fetch so as to 'actually fetch' the dataset and store it into the
        cache only if one needs it. The dataset may live on another machine, but
        maybe having access to the dataset from there is sufficient without
        having to copy the whole set"""

        src = self.src
        home = self.home()

        if not path.exists(src):
            raise IOError('"%s" does not exist' % src)
        else:
            skdata_dataset = path.join(home, self.image_dir_basename)
            if not path.exists(home):
                os.makedirs(home)
            if not path.exists(skdata_dataset):
                shutil.copytree(src, skdata_dataset, symlinks=False)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta()
    # ------------------------------------------------------------------------

    def meta(self):
        """extracts the meta data and assigns it to the '._meta' attribute"""
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        meta = []

        # -- get the list of original images
        suffix = path.join(self.image_dir_basename,
                           self.ORIGINAL_IMAGES_DIRNAME,
                           '*.png')

        pattern = self.home(suffix)

        img_filenames = sorted(glob(pattern))

        # -- get the list of training images
        trn_suffix = path.join(self.image_dir_basename,
                               self.TRAINING_IMAGES_DIRNAME,
                               '*.tif')

        trn_pattern = self.home(trn_suffix)

        trn_img_filenames = sorted(glob(trn_pattern))

        # -- compute the 'z' values for which we have annotations
        #    and store the path to the annotated image
        trn_z = {}
        for trn_img_filename in trn_img_filenames:

            img_basename = path.basename(trn_img_filename)
            int_pattern = r'I(\d+)_train.tif'
            match = re.match(int_pattern, img_basename)
            if match:
                z = int(match.groups()[0]) + 1
                trn_z[z] = trn_img_filename
            else:
                print 'no "z" value found for image %s' % trn_img_filename

        # -- getting the metadata
        for img_filename in img_filenames:

            # -- compute SHA1 hash of original image
            img_data = open(img_filename, 'rb').read()
            img_sha1 = hashlib.sha1(img_data).hexdigest()

            # -- infer 'z' value
            img_basename = path.basename(img_filename)
            int_pattern = r'z=(\d+).png'
            match = re.match(int_pattern, img_basename)
            if match:
                img_z = int(match.groups()[0])
            else:
                print 'no "z" value found for image %s' % img_filename
                img_z = None

            # -- see if we have some annotation for that image
            if img_z in trn_z.keys():
                anndl = [dict(type='binary_segmentation',
                              filename=trn_z[img_z],
                              basename=path.basename(trn_z[img_z]))]

            # -- getting image shape
            img_shape = imread(img_filename).shape

            #-- creating the metadata dictionnary for that image
            if img_z in trn_z.keys():
                data = dict(z=img_z,
                            filename=img_filename,
                            sha1=img_sha1,
                            shape=img_shape,
                            basename=img_basename,
                            annotation=anndl)
            else:
                data = dict(z=img_z,
                            filename=img_filename,
                            sha1=img_sha1,
                            shape=img_shape,
                            basename=img_basename)

            meta.append(data)

        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        """removes dataset cache"""
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Helper functions
    # ------------------------------------------------------------------------

    def get_annotation(self, annd):
        """returns the annotation for a given annotation dictionnary"""

        RGB_mask = (0, 255, 0)

        # -- reading the 'raw' annotated image
        arr = np.array(Image.open(annd['filename']))
        assert arr.ndim == 3, 'annotation is not RGB'

        # -- mask to extract the annotations from the training
        #    images. these annotations are curves drawn in green
        #    so the proper pixels to extract are such that the
        #    RGB value is (0, 255, 0)
        mask = np.zeros(arr.shape, dtype=arr.dtype)
        mask[:, :, 0] = RGB_mask[0]
        mask[:, :, 1] = RGB_mask[1]
        mask[:, :, 2] = RGB_mask[2]

        # -- binary mask (boolean view of the annotation)
        binary_mask = ((arr == mask).sum(2) == 3)

        # -- final float array to return
        annotations = np.empty(binary_mask.shape, dtype=np.float32)
        annotations[binary_mask] = 1.
        annotations[-binary_mask] = -1.

        return annotations


class Connectome(ConnectomeBase):
    TRAINING_IMAGES_DIRNAME = "trainingImages"
    ORIGINAL_IMAGES_DIRNAME = "originalAC3Images"
