import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import adjusted_rand_score

def rand_warp(Y_true, Y_pred,
              n_flips=10000, n_iterations=10, pixel_weight=10.0):

    assert Y_true.dtype == bool
    assert Y_pred.dtype == bool

    Y_true_lbl = ndi.label(Y_true)[0].ravel()

    min_loss = np.inf
    best_pixel = 0
    best_rand = 0
    best_Y_warp = Y_pred.copy()
    for i in xrange(n_iterations):
        print i
        Y_warp = best_Y_warp.copy()
        if i > 0:
            ridx = np.random.permutation(Y_warp.size)[:n_flips]
            Y_warp.flat[ridx] = Y_true.flat[ridx]
        Y_warp_lbl = ndi.label(Y_warp)[0].ravel()

        rand_score = adjusted_rand_score(Y_warp_lbl, Y_true_lbl)
        pixel_score = (Y_warp == Y_pred).mean()

        print pixel_score, rand_score
        loss = pixel_weight * pixel_score - rand_score
        print 'loss:', loss, 'min_loss:', min_loss, best_pixel, best_rand

        if loss < min_loss:
            min_loss = loss
            best_pixel = pixel_score
            best_rand = rand_score
            best_Y_warp = Y_warp.copy()

    return best_Y_warp



if __name__ == '__main__':
    from skimage import io
    io.use_plugin("freeimage")

    Y_true = io.imread("Y_true.tif")
    Y_pred = io.imread("Y_pred.tif")

    Y_warp = rand_warp(Y_true>0, Y_pred>0.5)

    import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')


