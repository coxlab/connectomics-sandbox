from skimage import io
io.use_plugin('freeimage')

from scipy import misc

from bangmetric import *
from bangmetric.isbi12 import *

gt = misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-labels.tif-29.png', flatten=True)
gv = misc.imread('/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png', flatten=True)

print rand_error(gt, gv)
print pixel_error(gt, gv)
print warp_error(gt, gv)
