# Authors : Nicolas Poilvert, <nicolas.poilvert@gmail.com>
# Licence : BSD 3-clause

import numpy as np
from numpy.random import permutation
from sthor.util.pad import filter_pad2d

from skimage.util.shape import view_as_windows


def get_trn_coords_labels(gt_img, fb,
                          fraction=1.):

    assert gt_img.ndim == 2
    h, w = gt_img.shape
    assert fb.ndim == 3
    fbh, fbw, fbn = fb.shape
    assert fraction > 0.

    epsilon = 1e-6

    # -- first we extract the H, W coordinates of the non-zero
    #    elements of ``gt_img``, i.e. the "membrane" pixels or
    #    "active patches"
    mask = (gt_img > 0.)
    H_nonzero, W_nonzero = mask.nonzero()

    # -- H, W coordinates of "background" pixels or "inactive
    #    patches"
    H_zero, W_zero = (-mask).nonzero()

    assert (H_nonzero.size + H_zero.size) == gt_img.size
    assert (W_nonzero.size + W_zero.size) == gt_img.size

    # -- extracting the membrane patches into a matrix of shape
    #    [n_active_patches, patch_dimension]
    padded_gt_img = filter_pad2d(gt_img[..., np.newaxis], (fbh, fbw)).squeeze()
    active_patch = view_as_windows(padded_gt_img, (fbh, fbw))
    assert active_patch.shape == (h, w, fbh, fbw)
    active_patch = active_patch.reshape((h, w, fbh * fbw))
    active_patch = active_patch[H_nonzero, W_nonzero]

    # -- normalizing the patches (zero mean, unit L2-norm)
    active_patch -= active_patch.mean(axis=1)[..., np.newaxis]
    lengths = np.sqrt((active_patch ** 2).sum(axis=1))
    lengths = np.where(lengths <= epsilon, 1., lengths)
    active_patch /= lengths[..., np.newaxis]

    # -- normalizing the filter bank
    fb = fb.reshape((fbh * fbw, fbn))
    fb -= fb.mean(axis=0)[np.newaxis, ...]
    fb /= np.sqrt((fb ** 2).sum(axis=0))[np.newaxis, ...]

    # -- response array of the patches to every filter
    responses = np.dot(active_patch, fb)

    # -- winners for every patch
    winners = responses.argmax(axis=1)
    assert (np.unique(winners) == np.arange(fbn)).all()

    # -- we compute the number of winning patches per filter
    n_winner_per_filter = np.empty((fbn,), dtype=np.int)
    for i in xrange(fbn):
        n_winner_per_filter[i] = (winners == i).sum()

    # -- finding out what the minimal number of winning patches is
    n_samples = n_winner_per_filter.min()
    n_background = min(int(fraction * n_samples), H_zero.size)
    assert n_samples > 1
    assert n_background > 1

    # -- extracting the proper H, W coordinates of the winning
    #    pixels for each filter (with an appropriate amount of
    #    "background" pixels)
    output = []
    for idx in xrange(fbn):

        f_mask = (winners == idx)

        # -- membrane pixels

        f_H = H_nonzero[f_mask]
        f_W = W_nonzero[f_mask]

        indices = permutation(np.arange(f_H.size))[:n_samples]

        f_H_out = f_H[indices]
        f_W_out = f_W[indices]
        labels_out = np.ones((n_samples,), dtype=gt_img.dtype)

        # -- background pixels

        indices0 = permutation(np.arange(H_zero.size))[:n_background]

        f_H0_out = H_zero[indices0]
        f_W0_out = W_zero[indices0]
        labels0_out = -np.ones((n_background,), dtype=gt_img.dtype)

        # -- merging the two together

        final_idx = permutation(np.arange(n_samples + n_background))
        H_out = np.concatenate((f_H_out, f_H0_out))
        W_out = np.concatenate((f_W_out, f_W0_out))
        labels_out = np.concatenate((labels_out, labels0_out))

        # -- add to output list
        output += [(H_out[final_idx], W_out[final_idx], labels_out[final_idx])]

    return output


def predict(X, clf_b):

    n_samples, n_features = X.shape
    n_clf = len(clf_b)

    out = np.empty((n_samples, n_clf), dtype=X.dtype)

    for idx in xrange(n_clf):
        out[:, idx] = clf_b[idx].decision_function(X)

    return out
