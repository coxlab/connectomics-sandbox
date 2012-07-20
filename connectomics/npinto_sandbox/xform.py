from scipy import ndimage
import numpy as np

def water(arr, sigma=1):

    yy, xx = np.indices(arr.shape)

    #sigma = np.minimum(0.2 * i + 1, l.shape[0])
    yyr, xxr = np.random.normal(0, sigma, size=((2,) + arr.shape))
    yyrg = ndimage.gaussian_filter(yyr, 2)
    xxrg = ndimage.gaussian_filter(xxr, 2)
    xxo = xx + xxrg
    xxo = xxo.astype(int).clip(0, arr.shape[1] - 1)
    yyo = yy + yyrg
    yyo = yyo.astype(int).clip(0, arr.shape[1] - 1)
    out = arr[yyo, xxo]

    return out
