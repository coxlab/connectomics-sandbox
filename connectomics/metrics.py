# author: Nicolas Poilvert
# license: BSD

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.integrate import trapz as trapezoidal
from scipy.ndimage import gaussian_filter

# -- spread of Gaussian filter for blurring the
#    Ground truth annotations
SIGMA = 1.

# -- epsilon is used to binarize the Gaussian
#    filtered ground truth annotations
EPSILON = 0.2


def _preprocess(arr, ref, force_rescale=True):
    """
    This function checks for the following properties
    in arrays ``arr`` and ``ref``:

    - ``arr`` and ``ref`` elements must be float32
      and if not these are transformed.
    - ``arr`` and ``ref`` are array-like objects, and
      if not, these are cast into numpy arrays.
    - ``arr`` and ``ref`` have the same shape.
    - ``arr`` and ``ref`` element values are bounded
      within [-1., 1.], and if not the array elements
      are rescaled to this interval.
    - elements of ``ref`` are also constrained to be
      given by either -1 or +1.

    then the arrays are raveled into 1D arrays and
    returned.

    Parameters
    ==========
    arr: array-like
        input array

    ref: array-like
        reference input array

    force_rescale: bool
        force the rescaling of the input array and of
        the ground truth array to [-1, 1]

    Returns
    =======
    arr_rescaled, ref_rescaled: 1D arrays of type 'float32'
        these arrays are rescaled to [-1, 1]
    """

    # -- cast to numpy arrays of type 'float32'
    arr_in = np.array(arr).astype(np.float32)
    ref_in = np.array(ref).astype(np.float32)

    # -- arrays must be of the same shape
    assert arr_in.shape == ref_in.shape

    # -- extract minimal and maximal values for both arrays
    arr_in_min, arr_in_max = arr_in.min(), arr_in.max()
    ref_in_min, ref_in_max = ref_in.min(), ref_in.max()

    # -- if element values are not within the interval
    #    [-1, 1], then the arrays are rescaled
    if arr_in_min == arr_in_max:
        arr_rescaled = np.zeros(arr_in.shape, dtype=np.float32)
    elif (arr_in_min < -1.) or (1. < arr_in_max):
        arr_rescaled = 1. - \
                      (2. / (arr_in_max - arr_in_min)) * \
                      (arr_in_max - arr_in)
    elif force_rescale:
        arr_rescaled = 1. - \
                      (2. / (arr_in_max - arr_in_min)) * \
                      (arr_in_max - arr_in)
    else:
        arr_rescaled = arr_in

    if ref_in_min == ref_in_max:
        ref_rescaled = np.zeros(ref_in.shape, dtype=np.float32)
    elif (ref_in_min < -1.) or (1. < ref_in_max):
        ref_rescaled = 1. - \
                      (2. / (ref_in_max - ref_in_min)) * \
                      (ref_in_max - ref_in)
    elif force_rescale:
        ref_rescaled = 1. - \
                      (2. / (ref_in_max - ref_in_min)) * \
                      (ref_in_max - ref_in)
    else:
        ref_rescaled = ref_in

    assert arr_rescaled.size == ref_rescaled.size

    # -- the reference array is supposed to be a binary map
    ref_rescaled = np.where(ref_rescaled < 0., -1., 1.)

    # -- applying a small Gaussian filter to "spread out" the
    #    ground truth annotation
    gref_rescaled = gaussian_filter(ref_rescaled, SIGMA)

    # -- binarizing the filtered ground truth
    tgref_rescaled = np.where(gref_rescaled > -1. + EPSILON, 1., -1.)

    return (arr_rescaled.ravel(), tgref_rescaled.ravel())


def rmse(arr, ref):
    """
    root mean square error between ``arr`` and ``ref``.
    The preprocessing step makes sure that the array
    elements are bounded withing [-1, 1]

    Parameters
    ==========
    arr: array-like
        input array

    ref: array-like
        reference input array

    Returns
    =======
    rmse: float
        Root mean square error
    """

    arr_in, ref_in = _preprocess(arr, ref)

    nitems_inv = 1. / arr_in.size
    rmse = 0.5 * np.sqrt(nitems_inv * norm(arr_in - ref_in) ** 2)

    return rmse


def brmse(arr, ref):
    """
    balanced root mean square error between ``arr`` and
    ``ref``. The preprocessing step makes sure that the
    array elements are bounded withing [-1, 1]

    Parameters
    ==========
    arr: array-like
        input array

    ref: array-like
        reference input array

    Returns
    =======
    brmse: float
        Balanced Root mean square error
    """

    arr_in, ref_in = _preprocess(arr, ref)

    # -- mask indicating the position of the positive
    #    elements in "ref"
    pmask = (ref_in > 0.)
    npos = pmask.sum()
    pitems_inv = 1. / npos

    # -- mask indicating the position of the negative
    #    elements in "ref"
    nmask = (ref_in < 0.)
    nneg = nmask.sum()
    nitems_inv = 1. / nneg

    brmse = 0.25 * np.sqrt(pitems_inv *
                           norm(arr_in[pmask] - ref_in[pmask]) ** 2) + \
            0.25 * np.sqrt(nitems_inv *
                           norm(arr_in[nmask] - ref_in[nmask]) ** 2)

    return brmse


def pearson(arr, ref):
    """
    computes the Pearson correlation coefficiant between the two
    array (considering the raveled arrays as two datasets of random
    variables with a given distribution).
    The preprocessing step ensures that the array elements are all
    within [-1, 1]

    Parameters
    ==========
    arr: array-like
        input array

    ref: array-like
        reference input array

    Returns
    =======
    pearson: float
        Pearson's correlation coefficiant
    """

    arr_in, ref_in = _preprocess(arr, ref)
    pearson = pearsonr(arr_in, ref_in)[0]

    return pearson


def ap(arr, ref, eps=0.001):
    """
    Computes many metrics related to precision, recall, accuracy
    and mean average precision
    """

    arr_in, ref_in = _preprocess(arr, ref)

    size = arr_in.size

    # -- indices that sort the array elements for the predicted
    #    target map
    si = arr_in.argsort()

    # -- now we sort both arrays
    gv = np.ones(size, dtype=np.float32)
    gt = ref_in[si]

    # -- initial values for the "True positives", "False positives"
    #    "True negatives" and "False negatives" when the threshold
    #    is lowest (and so all predicted target map values are clipped
    #    to +1)
    tp0 = (gv[gt > 0.] == 1.).sum()
    fp0 = (gv[gt < 0.] == 1.).sum()
    tn0 = (gv[gt < 0.] == -1.).sum()
    fn0 = (gv[gt > 0.] == -1.).sum()

    # -- now we sweep the threshold values up and recompute the
    #    TP, FP, TN and FN. For this we use the trick of cumulative
    #    sums
    tp = tp0 * np.ones(size, dtype=np.float32)
    fp = fp0 * np.ones(size, dtype=np.float32)
    tn = tn0 * np.ones(size, dtype=np.float32)
    fn = fn0 * np.ones(size, dtype=np.float32)

    tp[1:] -= (gt[:-1] > 0.).cumsum()
    fn[1:] += (gt[:-1] > 0.).cumsum()

    tn[1:] += (gt[:-1] < 0.).cumsum()
    fp[1:] -= (gt[:-1] < 0.).cumsum()

    # -- we add a little epsilon to every array so as to be able
    #    to compute derived quantities like recall, precision and
    #    accuracy without encountering a zero division error
    tp += eps
    fp += eps
    tn += eps
    fn += eps

    # -- precision, recall and accuracy
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # -- computing "exact" average precision
    recsi = recall.argsort()
    ap = trapezoidal(precision[recsi], recall[recsi])

    # -- computing best accuracy
    max_accuracy = accuracy.max()

    return ap, max_accuracy
