# IPython log file

logreg.fit(val_Y_pred, val_Y.ravel()>0)
Y_pred1 = mdl1.predict(tst_X)
Y_pred2 = logreg.predict(Y_pred1)
Y_pred1.shape
Y_pred2 = logreg.predict(Y_pred1[..., 0])
Y_pred2 = logreg.predict(view_as_windows(filter_pad2d(Y_pred1, (11, 11)), (11, 11, 1)).reshape(-1, 11**1.))
Y_pred2 = logreg.predict(view_as_windows(filter_pad2d(Y_pred1, (11, 11)), (11, 11, 1)).reshape(-1, 11**2.))
Y_pred2.shape
Y_pred2.reshape(512, 512)
Y_pred2.reshape(512, 512).shape
Y_pred2 = Y_pred2.reshape(512, 512)
Y_pred1
Y_pred2 = logreg.transform(view_as_windows(filter_pad2d(Y_pred1, (11, 11)), (11, 11, 1)).reshape(-1, 11**2.))
Y_pred2 = logreg.transform(view_as_windows(filter_pad2d(Y_pred1, (11, 11)), (11, 11, 1)).reshape(-1, 11**2.)).reshape(512, 512)
Y_pred1.shape
from skimage import io
io.use_plugin('freeimage')
Y_pred1 -= Y_pred1.min()
Y_pred1 /= Y_pred1.max()
Y_pred2 -= Y_pred2.min()
Y_pred2 /= Y_pred2.max()
Y_pred1
Y_pred2
Y_pred2.shape
io.imsave("Y_pred1.tif", Y_pred1[..., 0], plugin='freeimage')
io.imsave("Y_pred2.tif", Y_pred2, plugin='freeimage')
io.imsave("Y_pred2.tif", Y_pred2[32:-32, 32:-32], plugin='freeimage')
io.imsave("Y_pred1.tif", Y_pred1[32:-32, 32:-32, 0], plugin='freeimage')
Y_true = tst_Y.copy()
io.imsave("Y_true.tif", Y_true[32:-32, 32:-32], plugin='freeimage')
io.imsave("Y_true.tif", Y_true[64:-64, 64:-64], plugin='freeimage')
io.imsave("Y_pred1.tif", Y_pred1[64:-64, 64:-64, 0], plugin='freeimage')
io.imsave("Y_pred2.tif", Y_pred2[64:-64, 64:-64], plugin='freeimage')
get_ipython().magic(u'logstart ')
