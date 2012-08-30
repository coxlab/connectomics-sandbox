#!/usr/bin/env python

# -- base imports
import numpy as np
from numpy.random import permutation
from copy import deepcopy
import cPickle
from scaler import OnlineScaler
from skimage.filter import median_filter
from skimage.util.shape import view_as_windows
from pprint import pprint

# -- customs imports
from bangreadout.sqhinge import AverageLBFGSSqHingeClassifier as Classifier
from bangmetric.wildwest.isbi12 import warp2d
from sthor.model.slm import SequentialLayeredModel
from sthor.model.slm import _get_in_shape
from sthor.util.arraypad import pad
from sthor.operation.resample import resample

# -- if one wants to read the cascade description from a Pickle file, enter the
# name of the Pickle file here
CASCADE_PKL = 'inputs/top_7_7_7_px_bypass_mf_19_19_balanced.pkl'

# -- if we decide to use the warping of the annotations instead of the
# annotations themselves, then set the warping threshold here (see Fiji
# documentation to know what the threshold means, or play around with
# ``warp2d``)
WARP2D_THR = 0.2

# -- an example cascade description
custom_cascade = {
 'biases': None,
 'weights': None,
 'fuser': {'bypass_px': False,
           'concatenate_idx': [0, 1],
           'median_filter': True,
           'rolling_view_shape': (21, 21)},
 'cascade_desc':
 [
    {'desc':
       [
         [
           [
              [u'fbcorr',
                  {
                   u'initialize': {
                       u'filter_shape': [9, 9],
                       u'generate': [u'random:uniform', {u'rseed': 42}],
                       u'n_filters': 64
                       },
                   u'kwargs': {
                       u'max_out': None,
                       u'min_out': 0
                       }
                  }
              ],
              [u'lpool',
                  {
                   u'kwargs': {
                       u'ker_shape': [3, 3],
                       u'order': 1.0,
                       u'stride': 2
                       }
                  }
              ]
           ],
           [
              [u'fbcorr',
                  {
                   u'initialize': {
                       u'filter_shape': [5, 5],
                       u'generate': [u'random:uniform', {u'rseed': 42}],
                       u'n_filters': 128
                       },
                   u'kwargs': {
                       u'max_out': None,
                       u'min_out': 0
                       }
                  }
              ],
              [u'lpool',
                  {
                   u'kwargs': {
                       u'ker_shape': [3, 3],
                       u'order': 1.0,
                       u'stride': 2
                       }
                  }
              ]
           ]
         ]
       ],
   'sf_l': [[1.0, 2.0], [1.0, 2.0]],
   'use_warping': False,
   'balance_ratio': 0.55
    },
    {'desc':
       [
         [
           [
              [u'fbcorr',
                  {
                   u'initialize': {
                       u'filter_shape': [9, 9],
                       u'generate': [u'random:uniform', {u'rseed': 42}],
                       u'n_filters': 64
                       },
                   u'kwargs': {
                       u'max_out': None,
                       u'min_out': 0
                       }
                  }
              ],
              [u'lpool',
                  {
                   u'kwargs': {
                       u'ker_shape': [3, 3],
                       u'order': 1.0,
                       u'stride': 2
                       }
                  }
              ]
           ],
           [
              [u'fbcorr',
                  {
                   u'initialize': {
                       u'filter_shape': [5, 5],
                       u'generate': [u'random:uniform', {u'rseed': 42}],
                       u'n_filters': 128
                       },
                   u'kwargs': {
                       u'max_out': None,
                       u'min_out': 0
                       }
                  }
              ],
              [u'lpool',
                  {u'kwargs': {
                       u'ker_shape': [3, 3],
                       u'order': 1.0,
                       u'stride': 2
                       }
                  }
              ]
           ]
         ]
       ],
   'sf_l': [[1.0, 2.0], [1.0, 2.0]],
   'use_warping': False,
   'balance_ratio': None
    }
 ]
}


def model_transform(model_l, desc_l, sf_l, img):
    """
    This function will perform a multi-scale feature extraction with the
    same model (or stack of models) at every scale defined by the scaling
    factor list.

    Parameters
    ----------

    ``model_l``: list
        a list of SLM-type models (objects from the SLM model class)

    ``desc_l``: list
        a list giving the "description" of each model in ``model_l``

    ``sf_l``: list
        a list of floats defining the scaling factors to apply to the original
        image to downsample (or upsample) it

    ``img``: array-like
        2D or 3D input array for the models. shape is [h, w]

    Returns
    -------

    a 3D tensor representing the output of the model(s) when concatenated
    together. The shape will be [h, w, d] where ``d`` is the size of the feature
    vector for each input image pixels (in the first two dimensions).
    """

    assert img.ndim == 2 or img.ndim == 3

    h, w = img.shape[:2]
    if img.ndim == 3:
        depth = img.shape[-1]
    feature_map_l = []

    for idx, (model, desc, sf) in enumerate(zip(model_l, desc_l, sf_l)):

        print ' extracting features for model %3i of %3i' % (idx + 1,
                                                             len(model_l))

        for s in sf:

            print '  scaling factor %6.3f' % s

            # -- first we perform a spline interpolation of the image
            hp, wp = int(float(h) / float(s)), int(float(h) / float(s))
            if img.ndim == 2:
                simg = np.squeeze(resample(img[..., np.newaxis],
                                  (hp, wp, 1), order=3, intp2d=True))
            else:
                simg = resample(img, (hp, wp, depth), order=3, intp2d=True)

            # -- then we compute the ``in_shape`` (used for the padding)
            success, in_shape, out_shape = _get_in_shape(simg.shape[:2], desc,
                                                         interleave_stride=True)

            # -- we then pad the image with "mirror" effects
            h_pad = in_shape[0] - simg.shape[0]
            w_pad = in_shape[1] - simg.shape[1]
            h_pad_left, h_pad_right = (h_pad + 1) / 2, h_pad / 2
            w_pad_left, w_pad_right = (w_pad + 1) / 2, w_pad / 2

            if img.ndim == 2:
                simg = pad(simg, ((h_pad_left, h_pad_right),
                                  (w_pad_left, w_pad_right)),
                                  'symmetric')
            else:
                simg_tmp = np.empty((h_pad_left + hp + h_pad_right,
                                     w_pad_left + wp + w_pad_right,
                                     depth), dtype=img.dtype)
                for d in xrange(depth):
                    simg_tmp[..., d] = pad(simg[..., d],
                                          ((h_pad_left, h_pad_right),
                                           (w_pad_left, w_pad_right)),
                                           'symmetric')
                simg = simg_tmp

            # -- we process the padded image with the model
            X_trn = model.transform(simg, pad_apron=False,
                                    interleave_stride=True)

            # -- we now "upsample" to the final shape and add to the list
            # of feature maps
            d = X_trn.shape[-1]
            assert X_trn.dtype == np.float32
            feature_map_l += [resample(X_trn, (h, w, d), order=0, intp2d=False)]

    # -- finally we concatenate all the feature maps into one
    return np.dstack(feature_map_l)


def parse_cascade(desc_dict):
    """
    Extracts all the necessary informations about the cascade. The expected
    format of ``desc_dict`` is a dictionnary who's key-value pairs are :

    - ``cascade_desc`` a list of dictionnaries. Each of these specifies the
      description of a stack of models in the cascade (a stack can contain only
      one model of course)
    - ``fuser`` contains a dictionnary specifying the description of the fuser
      classifier (the classifier that will combine all the predictions for each
      stack of models in the cascade to produce a final prediction)
    - ``weights`` contains either nothing (None) or a list of 1D numpy arrays
      corresponding to the weight vectors of each classifier in the cascade,
      i.e. the classifiers at the "end" of each stack of models in the cascade
    - ``biases`` contains either nothing (None) or a list of floats
      corresponding to the biases values of each classifier in the cascade
    """

    # -- extracting the list that describes each stack of models in the cascade
    stacks_desc_l = desc_dict['cascade_desc']

    # -- computing the number of stacks
    n_stacks = len(stacks_desc_l)
    assert n_stacks >= 1

    # -- infering the number of cascades from the number of stacks
    n_cascades = n_stacks - 1
    assert n_cascades >= 0

    # -- extracting information concerning the "fuser" classifier
    fuser_desc = desc_dict['fuser']

    # -- some consistency tests
    assert len(fuser_desc['concatenate_idx']) <= n_stacks
    valid_indices = range(n_stacks)
    for idx in fuser_desc['concatenate_idx']:
        assert idx in valid_indices

    rvh, rvw = fuser_desc['rolling_view_shape']
    assert 1 <= rvh and rvh <= 100
    assert 1 <= rvw and rvw <= 100

    assert fuser_desc['bypass_px'] in [False, True]
    assert fuser_desc['median_filter'] in [False, True]

    if desc_dict['weights'] is None or len(desc_dict['weights']) == 0:
        weights = []
    else:
        weights = desc_dict['weights']

    if desc_dict['biases'] is None or len(desc_dict['biases']) == 0:
        biases = []
    else:
        biases = desc_dict['biases']

    return (n_cascades, n_stacks, stacks_desc_l, fuser_desc, weights, biases)


def process(tasks, *args):
    """
    ``tasks`` is supposed to be a list of image lists. Typically, one can expand
    a task as follows :

    for task in tasks:
        trn_X_l, trn_Y_l, tst_X_l, tst_Y_l = task

    A ``task`` typically corresponds to a cross-validation fold.

    ``trn_X_l`` is a list of training images. (typically 2D or 3D numpy arrays)
    ``trn_Y_l`` is a list of annotation images. (typically 2D numpy arrays)
    ``tst_X_l`` is a list of testing images (typically 2D numpy arrays)
    ``tst_Y_l`` is possibly empty (in which case, ``tst_X_l`` are "real" test
    images for which we have no corresponding annotation images)
    """

    # -- global parameters (randomize images and training vectors, how many
    # images to use to train the 'fuser')
    randomize = False
    n_fuser_trn_imgs = 1
    assert n_fuser_trn_imgs >= 1

    # -- getting the cascade complete information (from a pkl file or a custom
    # dictionnary). Modify the lines accordingly
    info_dict = cPickle.load(open(CASCADE_PKL, 'r'))
    #info_dict = custom_cascade
    info_dict['cascade_desc'][0]['balance_ratio'] = 0.55
    info_dict['cascade_desc'][1]['balance_ratio'] = 0.30
    info_dict['cascade_desc'][2]['balance_ratio'] = None

    n_cascades, n_stacks, stacks_desc_l, fuser_desc, weights, biases = \
            parse_cascade(info_dict)

    # -- determining whether we need to train the stacks of models in the
    # cascade by inspecting the ``weights`` and ``biases`` lists
    if len(weights) < n_stacks or len(biases) < n_stacks or \
       len(weights) > n_stacks or len(biases) > n_stacks:
        train_stacks = True
    else:
        train_stacks = False

    # -- making sure we have enough training images
    for task in tasks:
        trn_img_l, trn_gt_img_l, _, _ = task
        if train_stacks:
            assert len(trn_img_l) >= n_stacks + n_fuser_trn_imgs
            assert len(trn_img_l) == len(trn_gt_img_l)
        else:
            assert len(trn_img_l) >= n_fuser_trn_imgs
            assert len(trn_img_l) == len(trn_gt_img_l)

    # -- looping over the ``tasks``
    tst_pred, tst_gt = [], []

    for fold_idx, task in enumerate(tasks):

        print 'fold %2i of %2i' % (fold_idx + 1, len(tasks))

        # -- extracting the training and testing images for that
        # cross-validation fold
        trn_X_l, trn_Y_l, tst_X_l, tst_Y_l = task
        img_shape = trn_X_l[0].shape

        # -- splitting the training images into approximately equally sized
        # baskets for training each classifier in the cascade
        n_trn_imgs = len(trn_X_l)

        if randomize:
            order = permutation(range(n_trn_imgs)).tolist()
        else:
            order = range(n_trn_imgs)

        trn_X_ll, trn_Y_ll = [], []

        for i in range(n_stacks):
            trn_X_ll += [[trn_X_l[j].copy() for j
                in order[i:-n_fuser_trn_imgs:n_stacks]]]
            trn_Y_ll += [[trn_Y_l[j].copy() for j
                in order[i:-n_fuser_trn_imgs:n_stacks]]]

        fuser_trn_X_l = deepcopy(trn_X_l[-n_fuser_trn_imgs:])
        fuser_trn_Y_l = deepcopy(trn_Y_l[-n_fuser_trn_imgs:])

        # -- instantiating the stacks
        stacks = []
        for stacks_desc in stacks_desc_l:
            desc_l = stacks_desc['desc']
            sf_l = stacks_desc['sf_l']
            if 'use_warping' in stacks_desc.keys():
                use_warp = stacks_desc['use_warping']
            else:
                use_warp = False
            if 'balance_ratio' in stacks_desc.keys():
                balance_ratio = stacks_desc['balance_ratio']
            else:
                balance_ratio = None
            model_l = []
            n_features = 0
            for desc, sf in zip(desc_l, sf_l):
                success, in_shape, out_shape = _get_in_shape(img_shape, desc,
                        interleave_stride=True)
                if success:
                    model = SequentialLayeredModel(in_shape, desc)
                    n_features += len(sf) * model.n_features
                    model_l += [model]
                else:
                    raise ValueError('could not instantiate model because' +
                            ' of no appropriate padding available')
            stacks += [(model_l, desc_l, sf_l, n_features, use_warp,
                        balance_ratio)]

        # -- instantiating a classifier and a scaler for each stack in the
        # cascade
        classifiers = []
        for _, _, _, n_features, _, balance_ratio in stacks:
            classifiers += [(Classifier(n_features, negfrac=balance_ratio),
                             OnlineScaler(n_features))]

        # -- printing some information on screen
        print '----- STACKS DESCRIPTIONS -----'
        for idx, (_, desc_l, sf_l, n_features, use_warp, balance_ratio) \
                in enumerate(stacks):
            print ' STACK %2i in the cascade' % (idx + 1,)
            print ' dim of feature vector = %6i' % n_features
            print ' using warping of annotation = %s' % use_warp
            print ' using balancing ratio of : %s' % balance_ratio
            print ' models in the stack :'
            for desc in desc_l:
                pprint(desc)
            print ' scaling factors of each model in the stack :'
            for sf in sf_l:
                pprint(sf)
        print '----- END STACK DESCRIPTIONS -----'

        # -- training the classifiers associated with each stack in the cascade
        if train_stacks:
            for i, (trn_X_l, trn_Y_l) in enumerate(zip(trn_X_ll, trn_Y_ll)):

                c_model_l, c_desc_l, c_sf_l, c_n_features, \
                        c_use_warp, _ = stacks[i]
                c_clf, c_scaler = classifiers[i]

                for img, gt_img in zip(trn_X_l, trn_Y_l):

                    in_img = img.copy()
                    if i > 0:
                        for j in range(i):
                            model_l, desc_l, sf_l, n_features, _, _ = stacks[j]
                            clf, scaler = classifiers[j]
                            f_map = model_transform(model_l, desc_l, sf_l,
                                    in_img)
                            sf_map = scaler.transform(f_map.reshape(-1,
                                n_features))
                            in_img = clf.transform(sf_map).reshape(img_shape)
                            if c_use_warp:
                                in_img_for_warp = in_img.copy()
                            in_img -= in_img.mean()
                            in_img /= in_img.std()

                    f_map = model_transform(c_model_l, c_desc_l, c_sf_l, in_img)
                    sf_map = c_scaler.fit_transform(f_map.reshape(-1,
                        c_n_features))
                    if c_use_warp:
                        c_gt_img = warp2d((gt_img > 0.), in_img_for_warp,
                                threshold=WARP2D_THR).ravel()
                    else:
                        c_gt_img = (gt_img > 0.).ravel()
                    c_clf.partial_fit(sf_map, c_gt_img,
                            w_start=np.zeros(sf_map.shape[1], dtype=np.float32),
                            b_start=np.zeros(1, dtype=np.float32),
                            mini_batch_size=30000,
                            n_maxfun=10, bfgs_m=30)

        else:
            # -- if we don't have to train the stacks we just dump the values
            # for the weights and biases directly into the corresponding
            # attributes of the classifiers
            for clf_idx, (clf, _) in enumerate(classifiers):
                clf.W = weights[clf_idx]
                clf.b = biases[clf_idx]

        # -- training the 'fuser'
        bypass_px = fuser_desc['bypass_px']
        concatenate_idx = fuser_desc['concatenate_idx']
        rv_shape = fuser_desc['rolling_view_shape']
        fh, fw = rv_shape

        if bypass_px:
            fuser_n_features = (1 + len(concatenate_idx)) * np.product(rv_shape)
        else:
            fuser_n_features = len(concatenate_idx) * np.product(rv_shape)

        if fh % 2 == 0:
            h_left, h_right = fh / 2, fh / 2 - 1
        else:
            h_left, h_right = fh / 2, fh / 2
        if fw % 2 == 0:
            w_left, w_right = fw / 2, fw / 2 - 1
        else:
            w_left, w_right = fw / 2, fw / 2

        fuser = Classifier(fuser_n_features)

        for img, gt_img in zip(fuser_trn_X_l, fuser_trn_Y_l):

            in_img = img.copy()
            pred_to_concatenate = []
            for j in range(n_stacks):
                model_l, desc_l, sf_l, n_features, _, _ = stacks[j]
                clf, scaler = classifiers[j]
                f_map = model_transform(model_l, desc_l, sf_l, in_img)
                if train_stacks:
                    sf_map = scaler.transform(f_map.reshape(-1, n_features))
                else:
                    sf_map = scaler.fit_transform(f_map.reshape(-1, n_features))
                in_img = clf.transform(sf_map).reshape(img_shape)
                in_img -= in_img.mean()
                in_img /= in_img.std()
                if j in concatenate_idx:
                    pred_to_concatenate += [pad(in_img, ((h_left, h_right),
                        (w_left, w_right)), 'symmetric')]

            if bypass_px:
                pred_to_concatenate += [pad(img, ((h_left, h_right),
                    (w_left, w_right)), 'symmetric')]

            f_map = np.dstack(pred_to_concatenate)
            rf_map = view_as_windows(f_map, rv_shape + (f_map.shape[-1],))
            rf_map = np.ascontiguousarray(rf_map.reshape(-1, fuser_n_features))

            fuser.partial_fit(rf_map, (gt_img > 0.).ravel(),
                    w_start=np.zeros(rf_map.shape[1], dtype=np.float32),
                    b_start=np.zeros(1, dtype=np.float32),
                    mini_batch_size=30000,
                    n_maxfun=10, bfgs_m=30)

        # -- testing images
        use_median_filter = fuser_desc['median_filter']
        fold_pred, fold_gt_pred = [], []

        for tst_idx, img in enumerate(tst_X_l):

            in_img = img.copy()
            pred_to_concatenate = []

            for j in range(n_stacks):
                model_l, desc_l, sf_l, n_features, _, _ = stacks[j]
                clf, scaler = classifiers[j]
                f_map = model_transform(model_l, desc_l, sf_l, in_img)
                if train_stacks:
                    sf_map = scaler.transform(f_map.reshape(-1, n_features))
                else:
                    sf_map = scaler.fit_transform(f_map.reshape(-1, n_features))
                in_img = clf.transform(sf_map).reshape(img_shape)
                in_img -= in_img.mean()
                in_img /= in_img.std()
                if j in concatenate_idx:
                    pred_to_concatenate += [pad(in_img, ((h_left, h_right),
                        (w_left, w_right)), 'symmetric')]

            if bypass_px:
                pred_to_concatenate += [pad(img, ((h_left, h_right),
                    (w_left, w_right)), 'symmetric')]

            f_map = np.dstack(pred_to_concatenate)
            rf_map = view_as_windows(f_map, rv_shape + (f_map.shape[-1],))
            rf_map = np.ascontiguousarray(rf_map.reshape(-1, fuser_n_features))

            prediction = fuser.transform(rf_map).reshape(img_shape)

            if use_median_filter:
                prediction = median_filter(prediction, radius=2, percent=60)

            fold_pred += [prediction]

            if len(tst_Y_l) == len(tst_X_l):
                fold_gt_pred += [tst_Y_l[tst_idx]]

        # -- saving predicted images and possibly the Ground Truth (gt)
        tst_pred += [fold_pred]
        if len(tst_Y_l) == len(tst_X_l):
            tst_gt += [fold_gt_pred]

    # -- preparing the data to return
    n_folds = len(tst_pred)
    n_preds = len(tst_pred[0])
    h, w = tst_pred[0][0].shape
    output_pred = np.zeros((n_preds, h, w, n_folds), dtype=tst_pred[0][0].dtype)
    for fold_idx, fold in enumerate(tst_pred):
        for img_idx, img in enumerate(fold):
            output_pred[img_idx, ..., fold_idx] = deepcopy(img)
    if len(tst_gt) == len(tst_pred):
        h, w = tst_gt[0][0].shape
        output_true = np.zeros((n_preds, h, w, n_folds),
                dtype=tst_gt[0][0].dtype)
        for fold_idx, fold in enumerate(tst_gt):
            for img_idx, img in enumerate(fold):
                output_true[img_idx, ..., fold_idx] = deepcopy(img)
    else:
        output_true = []

    # -- if we trained the stacks/classifiers then we keep a copy of the
    # latest weights and biases
    if train_stacks:
        info_dict['weights'] = [clf.W.copy() for clf, _ in classifiers]
        info_dict['biases'] = [clf.b.copy() for clf, _ in classifiers]

    to_save = {'model_info': info_dict}

    return (output_true, output_pred, to_save)
