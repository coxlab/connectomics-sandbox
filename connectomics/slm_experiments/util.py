# Authors : Nicolas Poilvert, <nicolas.poilvert@gmail.com>
# Licence : BSD 3-clause

import numpy as np
from os import path
import cPickle
from glob import glob
from numpy.random import permutation

from skimage.shape import view_as_windows


def generate_Xy_train(farr,
                      h_size, w_size,
                      trn_idx, tst_idx,
                      tm,
                      memory_limit,
                      randomize):

    # -- transposing the feature array to get the
    #    height and width first, followed respectively
    #    by number of images, number of scales, and
    #    number of features per scale per image
    tfarr = np.transpose(farr, (2, 3, 0, 1, 4))

    # -- then we create a rolling view over the transposed
    #    feature array
    rtfarr = view_as_windows(tfarr, (h_size, w_size, 1, 1, 1))

    # -- number of "super-pixels" in the rolling view in h
    #    and w directions (these correspond to the size of the
    #    original image minus the aprons)
    hp, wp = rtfarr.shape[:2]

    # -- getting the (h, w) coordinates of the training examples
    trn_h, trn_w, _, _ = get_trn_tst_coords(hp, wp,
                                            trn_idx, tst_idx,
                                            randomize=randomize)

    # -- splitting the coordinate arrays in batches of
    #    smaller coordinate arrays for memory management
    feature_dimensionality = np.prod(np.array(rtfarr.shape[2:]))
    trn_h_l, trn_w_l = split_for_memory(trn_h, trn_w,
                                        feature_dimensionality,
                                        rtfarr.itemsize,
                                        memory_limit)

    # -- building the generator of X_train, y_train
    for h_arr, w_arr in zip(trn_h_l, trn_w_l):
        yield (rtfarr[h_arr, w_arr, ...].reshape(h_arr.size, -1),
               tm[h_arr, w_arr])


def generate_Xhw_test(farr,
                      h_size, w_size,
                      trn_idx, tst_idx,
                      memory_limit,
                      randomize):

    # -- transposing the feature array to get the
    #    height and width first, followed respectively
    #    by number of images, number of scales, and
    #    number of features per scale per image
    tfarr = np.transpose(farr, (2, 3, 0, 1, 4))

    # -- then we create a rolling view over the transposed
    #    feature array
    rtfarr = view_as_windows(tfarr, (h_size, w_size, 1, 1, 1))

    # -- number of "super-pixels" in the rolling view in h
    #    and w directions (these correspond to the size of the
    #    original image minus the aprons)
    hp, wp = rtfarr.shape[:2]

    # -- getting the (h, w) coordinates of the training examples
    _, _, tst_h, tst_w = get_trn_tst_coords(hp, wp,
                                            trn_idx, tst_idx,
                                            randomize=randomize)

    # -- splitting the coordinate arrays in batches of
    #    smaller coordinate arrays for memory management
    feature_dimensionality = np.prod(np.array(rtfarr.shape[2:]))
    tst_h_l, tst_w_l = split_for_memory(tst_h, tst_w,
                                        feature_dimensionality,
                                        rtfarr.itemsize,
                                        memory_limit)

    # -- building the generator
    for h_arr, w_arr in zip(tst_h_l, tst_w_l):
        yield (rtfarr[h_arr, w_arr, ...].reshape(h_arr.size, -1),
               h_arr, w_arr)


class OnlineScaler(object):

    def __init__(self, d):
        """initializes an Online Scaler object

        attributes
        ----------

        `r_ns`: int64
            running number of sample feature vectors

        `r_mn`: 1D array of float64
            running mean feature vector

        `r_va`: 1D array of float64
            running vector of feature-wise variance

        methods
        -------

        `partial_fit`:
            updates the values of `r_ns`, `r_mn` and `r_va`
            using the new data in the batch

        `transform`:
            returns a batch, where all the feature vectors have
            been translated uniformly by the current mean feature
            vector and normalized by the current standard deviation
            (i.e. the square root of the current variance) feature
            wise.

        `fit_transform`:
            first perform a partial fit of the Scaler and then
            transforms the batch

        conventions
        -----------
        'r' means 'running' like in 'running average'
        'ns' stands for 'number of samples'
        'mn' stands for 'mean'
        'va' stands for 'variance'
        'd' is the dimensionality of the feature vector
        """

        self.r_ns = np.int64(0)
        self.r_mn = np.zeros(d, dtype=np.float64)
        self.r_va = np.zeros(d, dtype=np.float64)
        self.d = np.int64(d)

    def partial_fit(self, X):

        # -- X has to be 2D
        if X.ndim != 2:
            raise ValueError('training batch must be 2D')

        # -- makes sure that the batch X has a second
        #    dimension equal to d
        if X.shape[1] != self.d:
            raise ValueError('wrong number of features')

        # -- makes sure that the batch has at least one
        #    sample
        if X.shape[0] < 1:
            raise ValueError('batch is too small')

        # -- extract number of samples in batch and
        #    computes the sum of X and X**2 of the batch
        b_ns = np.int64(X.shape[0])
        b_su = np.float64(X.sum(axis=0))
        b_sq = np.float64((X**2).sum(axis=0))

        # -- updating the values for the number of samples
        #    the mean vector X and the variance. 'c' stands
        #    for 'current'
        c_ns = self.r_ns
        c_mn = self.r_mn
        c_va = self.r_va

        self.r_ns = c_ns + b_ns
        self.r_mn = 1. / self.r_ns * (c_ns * c_mn + b_su)
        self.r_va = 1. / self.r_ns * (c_ns * (c_va + c_mn**2) + b_sq) \
                    - self.r_mn**2

    def transform(self, X):

        # -- returns (X - mean) / sqrt(var)
        meanX = (self.r_mn[np.newaxis, :]).astype(X.dtype)
        std = np.sqrt((self.r_va[np.newaxis, :]).astype(X.dtype))
        return (X - meanX) / std

    def fit_transform(self, X):

        # -- first update the attributes of the Scaler object
        self.partial_fit(X)

        # -- then transform X
        return self.transform(X)


def get_memmap_array(data_file_path):

    if not path.exists(data_file_path):
        raise ValueError('not a valid path')

    # -- getting the Pickle filename that goes with the data file
    pattern = data_file_path + '*.pkl'
    valid_files = glob(pattern)

    assert len(valid_files) == 1

    # -- extract the dictionnary in the Pickle file
    dico = cPickle.load(open(valid_files[0], 'r'))

    # -- we finally create the memmap array
    arr = np.memmap(data_file_path, dtype=dico['dtype'],
                    mode='r', shape=dico['global_memmap_array_shape'])

    return arr


def zero_mean_unit_variance(img):

    assert img.dtype == np.float32

    mean = img.mean()
    std = img.std()

    return (img - mean) / std


def get_reduced_tm(tm, h_size, w_size):

    assert h_size % 2 != 0
    assert w_size % 2 != 0

    assert h_size <= tm.shape[0]
    assert w_size <= tm.shape[1]

    h_start, h_stop = h_size / 2, tm.shape[0] - h_size / 2
    w_start, w_stop = w_size / 2, tm.shape[1] - w_size / 2

    new_tm = tm[h_start:h_stop, w_start:w_stop]

    return new_tm


def split_for_memory(h_arr, w_arr, feature_size, itemsize,
                     memory_limit):

    assert h_arr.size == w_arr.size

    bytes_in_arr = h_arr.size * itemsize * feature_size

    nsplits = 1 + bytes_in_arr / memory_limit
    blk_coords = get_block_coords1D(h_arr.size, nsplits)
    h_list = [h_arr[blk[0]:blk[1]] for blk in blk_coords]
    w_list = [w_arr[blk[0]:blk[1]] for blk in blk_coords]

    return h_list, w_list


def normalize_feature_map(fmap):

    assert fmap.ndim == 3

    mean = fmap.reshape(-1, fmap.shape[-1]).mean(axis=0)
    zmean_fmap = fmap - mean[np.newaxis, np.newaxis, :]
    std = fmap.reshape(-1, fmap.shape[-1]).std(axis=0)
    zmean_unitvar_fmap = zmean_fmap / std[np.newaxis, np.newaxis, :]

    return zmean_unitvar_fmap


def get_trn_tst_coords(height, width,
                       trn_idx, tst_idx,
                       randomize=False):

    assert len(trn_idx) == len(tst_idx)
    assert len(trn_idx) <= height * width
    assert hasattr(trn_idx, '__iter__')
    assert hasattr(tst_idx, '__iter__')

    trn_idx = np.array(trn_idx)
    tst_idx = np.array(tst_idx)

    # -- variables
    nblks = trn_idx.size
    nitems = height * width

    # -- coordinates of all the points in a matrix
    #    of size (height, width)
    W, H = np.meshgrid(range(width), range(height))
    h_coords = H.flatten()
    w_coords = W.flatten()

    # -- get block coordinates
    blk_coords = np.array(get_block_coords1D(nitems, nblks))

    # -- train and test boolean masks
    trn_mask = np.ones(nitems, dtype=bool)
    for a_min, a_max in blk_coords[tst_idx]:
        trn_mask[a_min:a_max] = False
    tst_mask = - trn_mask

    # -- computing the train and test coordinates
    trn_h, tst_h = h_coords[trn_mask], h_coords[tst_mask]
    trn_w, tst_w = w_coords[trn_mask], w_coords[tst_mask]

    # -- randomizing the coordinates if needed
    if randomize:
        trn_permutation = permutation(trn_h.size)
        tst_permutation = permutation(tst_h.size)
        trn_h, trn_w = trn_h[trn_permutation], trn_w[trn_permutation]
        tst_h, tst_w = tst_h[tst_permutation], tst_w[tst_permutation]

    return (trn_h, trn_w, tst_h, tst_w)


def get_block_coords1D(length, nblk):

    assert length >= nblk

    # -- length of internal regular blocks
    blk_length = int(length) / int(nblk)

    # -- lowest and highest abscissa for the blocks
    a_start = [i * blk_length for i in xrange(nblk)]
    a_stop = [i * blk_length for i in xrange(1, nblk) if nblk > 1] \
              + [length]

    # -- building the block coordinates
    blk_coords = []
    for a_min, a_max in zip(a_start, a_stop):
        blk_coords += [(a_min, a_max)]

    return blk_coords


def get_block_coords2D(height, width, nblk_h, nblk_w):

    assert height >= nblk_h
    assert width >= nblk_w

    # -- shape of internal regular blocks
    blk_height = int(height) / int(nblk_h)
    blk_width = int(width) / int(nblk_w)

    # -- lowest and highest height values for the block
    h_start = [i * blk_height for i in xrange(nblk_h)]
    h_stop = [i * blk_height for i in xrange(1, nblk_h) if nblk_h > 1] \
              + [height]

    # -- lowest and highest width values for the blocks
    w_start = [i * blk_width for i in xrange(nblk_w)]
    w_stop = [i * blk_width for i in xrange(1, nblk_w) if nblk_w > 1] \
              + [width]

    # -- building the block coordinates
    blk_coords = []
    for h_min, h_max in zip(h_start, h_stop):
        for w_min, w_max in zip(w_start, w_stop):
            blk_coords += [(h_min, h_max, w_min, w_max)]

    return blk_coords
