# IPython log file

Y_pred.shape
Y_pred.reshape(512, 512)
io.imsave('tmp.tif', Y_pred)
io.imsave('tmp.tif', Y_pred.reshape(512, 512))
get_ipython().magic(u'logstart ')
