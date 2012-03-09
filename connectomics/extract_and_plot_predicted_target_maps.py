#!/usr/bin/env python

"""
this script takes the predicted target maps  from a
Pickle file, along with the 'true' target maps from
the  connectome dataset. Then,  for each target map
it dumps  an image file  with  the 'predicted'  and
'true' target maps shown side-by-side.
"""

import os
import shutil
from os import path
import cPickle
import scipy as sp
import numpy as np
from subprocess import call

# -- for logging
import logging as log
log.basicConfig(level=log.INFO)

# -- need the connectome dataset object
from coxlabdata.connectome import ConnectomicsHP as Connectome
from parameters import DATASET_PATH
from parameters import IM_SIZE

# -- for drawing images and text
from PIL import Image, ImageDraw, ImageFont

# -- for computing the metrics
import metrics

#--------------
# Main function
#--------------

from optparse import OptionParser, SUPPRESS_HELP

usage = """usage: python program.py [options] <tms_pkl> <tms_image_dir>

    tms_pkl
        name(s) of the Pickle file(s) containing the raw predicted
        target maps, i.e. the array containing the values of the
        classifier's decision function at every pixel. Many pkl
        filenames can be given
    tms_image_dir
        name of the directory in which the target map
        images should be saved for visualization

[options]:
    -t, --threshold
        once rescaled between -1 and 1, all predicted
        target map values *below* this threshold are
        clipped to -1
        [default -1.0]

    -o, --overwrite
        if the target map image directory already exists
        should the program overwrite it
        [default False]"""


def main():
    """front-end with command-line options"""

    parser = OptionParser(add_help_option=False)

    parser.add_option('-t', '--threshold',
                      dest='thr',
                      type='float',
                      default=-1.0,
                      help=SUPPRESS_HELP)

    parser.add_option('-o', '--overwrite',
                      dest='overwrite',
                      default=False,
                      action='store_true',
                      help=SUPPRESS_HELP)

    parser.add_option('-h', '--help',
                      dest='help',
                      default=False,
                      action='store_true',
                      help=SUPPRESS_HELP)

    (options, args) = parser.parse_args()

    if len(args) < 2:
        print usage
        return
    elif options.help:
        print usage

    # -- creating the directory to contain the predicted target
    #    map images if it does not exist already
    save_dir = args[-1]
    if not path.exists(save_dir):
        os.makedirs(save_dir)
    elif options.overwrite:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        print '%s already exist!' % save_dir
        return

    # -- only considers the pickle files that actually exist
    tm_pkl_fnames = []
    for filename in args[:-1]:
        if path.exists(filename):
            tm_pkl_fnames += [filename]
        else:
            log.info('%s does not exist. ignoring' % filename)

    # -- some parameters of the program

    # number of pickle files to process
    n_files = len(tm_pkl_fnames)

    # pixel size of unit square side used to plot the final
    # target map images
    # +--------------+
    # |    |    |    | -> basic block is (square_size, square_size)
    # |----+----+----|    in shape
    # |    |    |    |
    # +--------------+
    square_side = IM_SIZE

    # height and width of final target map images, i.e. the global
    # shape of the image described above
    height, width = 2 * square_side, square_side * (1 + (n_files + 1) / 2)

    # -- opening Pickle file(s) containing the predicted target maps
    #    (each pickle contains as many predicted target maps as there
    #     are images in the training set)
    pred_tms = []
    for tm_pkl_fname in tm_pkl_fnames:
        log.info('loading raw target maps from %s' % tm_pkl_fname)
        pred_tms += [cPickle.load(open(tm_pkl_fname, 'rb'))]

    # -- loading the 'true' target maps and the original images converted
    #    to grayscale. Both the target maps and the grayscale original
    #    images are rescaled to [-1, +1] such that the sub-images in the
    #    final images are all displayed in the same range of pixel values
    connectome_obj = Connectome(DATASET_PATH)
    meta = connectome_obj.meta()
    annotated_meta = sorted([imgd for imgd in meta if 'annotation' in imgd])
    gt_tms = []
    for imgd in annotated_meta:
        binary_tm= [connectome_obj.get_annotation(annd) for annd in
                    imgd['annotation']]
        gt_tms += [np.where(btm == False, -1., 1.).astype(np.float32)
                    for btm in binary_tm]

    image_fnames = sorted([imgd['filename'] for imgd in annotated_meta])
    raw_original_images = [
            sp.misc.imread(fname, flatten=True).astype(np.float32)
            for fname in image_fnames
            ]
    original_images = [normalize(arr) for arr in raw_original_images]

    assert len(original_images) == len(gt_tms)

    # -- for each list of "raw" target maps for each pickle file, we
    #    first normalize the target maps between -1 and +1 and then
    #    we threshold these normalized maps
    final_pred_tms_per_pkl_file = [normalize_and_threshold(tms, options.thr)
                                   for tms in pred_tms]

    # -- for each training image with its associated "ground truth"
    #    target map, we compute a set of metrics by comparing to the
    #    "given" target map (and we do this for every pkl file, which
    #    corresponds to one model)
    tm_metrics = []
    for idx, gt_tm in enumerate(gt_tms):
        metrics_per_image = []
        for gv_tms in final_pred_tms_per_pkl_file:
            gv_tm = gv_tms[idx]
            reduced_gt_tm = downscale_tm(gt_tm, gv_tm)
            pearson = metrics.pearson(gv_tm, reduced_gt_tm)
            ap, _ = metrics.ap(gv_tm, reduced_gt_tm)
            metrics_per_image += [dict(pearson=pearson,
                                       ap=ap)]
        tm_metrics.append(metrics_per_image)

    # -- time to dump the target maps to file
    for idx, (image, gt_tm)  in enumerate(zip(original_images, gt_tms)):

        image_basename = path.basename(image_fnames[idx])
        log.info('dumping target maps for %s' % image_basename)

        # dimensions of the final image array
        assert image.shape[:2] == gt_tm.shape[:2]
        assert image.shape[0] == image.shape[1]
        length = image.shape[0]
        h, w = 2 * length, length * (1 + (n_files + 1) / 2)

        # including the original image and the ground truth
        # target map in the image array
        image_array = np.zeros((h, w), dtype=np.float)
        image_array[:length, :length] = image
        image_array[length:, :length] = gt_tm

        # now including all the final target maps for the
        # considered image, for all the pickle files
        for pkl_idx, tms in enumerate(final_pred_tms_per_pkl_file):
            tm = tms[idx]
            h_start, h_stop = pkl_idx % 2 * length, \
                              (pkl_idx % 2 + 1) * length
            w_start, w_stop = (pkl_idx / 2 + 1)* length, \
                              (pkl_idx / 2 + 2)* length
            image_array[h_start:h_stop, w_start:w_stop] = \
                                       upscale_tm(tm, gt_tm)

        # inserting text on image
        image_rgb = arr_to_rgb_obj(image_array)
        for i in xrange(len(final_pred_tms_per_pkl_file)):
            anchor = ((i / 2 + 1)* length, i % 2 * length)
            strings = [path.basename(tm_pkl_fnames[i])]
            for key, value in tm_metrics[idx][i].items():
                strings += ["%s = %4.2f" % (key, value)]
            insert_text(strings, image_rgb, anchor)

        # saving the array as an image
        filename = image_basename + '_thr_%s_.png' % options.thr
        image_path = path.join(save_dir, filename)
        image_rgb.save(image_path)

        # resizing the image to the desired final shape
        final_size = '%sx%s' % (height, width)
        call(["/usr/bin/convert",
              image_path,
              "-resize", final_size,
              image_path])

    return

#------------------
# Utility functions
#------------------

def downscale_tm(gt_tm, gv_tm):
    """returns the `gt_tm` without the aprons. Aprons are
    infered from the shape of `gv_tm`"""

    h_apron = (gt_tm.shape[0] - gv_tm.shape[0]) / 2
    w_apron = (gt_tm.shape[1] - gv_tm.shape[1]) / 2
    h, w = gt_tm.shape[0], gt_tm.shape[1]

    return gt_tm[h_apron:h-h_apron, w_apron:w-w_apron]


def upscale_tm(gv_tm, gt_tm):
    """returns `gv_tm` but with some filled values from the
    apron regions. Aprons are infered from the shape of
    `gt_tm`"""

    h_apron = (gt_tm.shape[0] - gv_tm.shape[0]) / 2
    w_apron = (gt_tm.shape[1] - gv_tm.shape[1]) / 2
    h, w = gt_tm.shape[0], gt_tm.shape[1]

    gv_tm_mean = gv_tm.mean()
    final_gv_tm = gv_tm_mean * np.ones(gt_tm.shape, dtype=gv_tm.dtype)
    final_gv_tm[h_apron:h-h_apron, w_apron:w-w_apron] = gv_tm

    return final_gv_tm


def normalize_and_threshold(tms, thr):
    """Normalizes the raw target maps to the interval
    [-1, +1] and then applies a threshold"""

    #-- loop over target maps
    final_tms = []
    for tm_gv in tms:

        # -- filtering of the Given target map such that
        #    the target map is first rescale between -1 and 1
        #    and then we suppress all rescaled prediction
        #    values below a given threshold

        # -- rescaling to [-1, 1]
        final_tm_gv = normalize(tm_gv, low=-1., high=1.)

        # -- cutoff threshold
        final_tm_gv = np.where(final_tm_gv < thr, -1., final_tm_gv)

        # -- saving normalized and thresholded target map
        final_tms += [final_tm_gv]

    return final_tms


def normalize(arr_in, low=-1., high=1.):
    """normalizes an array (element-wise) to another interval"""

    arr = np.array(arr_in).astype(np.float32)

    minimum, maximum = arr.min(), arr.max()

    if maximum - minimum > 0.:
        arr_out = low + ((high - low) / (maximum - minimum)) * \
                  (arr - minimum)

    return arr_out


def arr_to_rgb_obj(arr_in):
    """transforms a float array into a proper 'RGB' PIL image object"""

    # -- we rescale the array between 0. and 255. (this is necessary
    #    for the conversion to 'RGB' mode)
    scaled_arr = normalize(arr_in, low=0., high=255.)

    # -- 'raw' (as in 'float') PIL image object
    obj = Image.fromarray(scaled_arr)

    return obj.convert(mode="RGB")


def insert_text(strings, img_obj, anchor,
                color=(245, 245, 220),
                size=25):
    """
    this function inserts lines of text derived
    from a list 'strings', starting at a position
    given by the tuple 'anchor' (which is read as
    (width, height)) on the PIL image object
    'img_obj'
    """

    # -- selection of the font type for displaying the text
    font = ImageFont.truetype('/usr/share/fonts/TTF/luximbi.ttf', size=size)

    # -- size of the text 'box'
    height = font.getsize(strings[0])[1]
    width = max([font.getsize(string)[0] for string in strings])

    # -- creating a new image object that contains the text
    txt = Image.new('RGBA', (width, height * len(strings)),
                    (255, 255, 255, 1))
    d = ImageDraw.Draw(txt)
    for i, string in enumerate(strings):
        d.text((0, i * height), string, font=font, fill=color)

    # -- inserting the text image object into the input image object
    img_obj.paste(txt, anchor, txt)

    return


if __name__ == "__main__":
    main()
