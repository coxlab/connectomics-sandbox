from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import convnet
import arraypad
from scipy import misc

#img = data.camera()
img = convnet.get_X_Y()[0]
#img = arraypad.pad(img, 512, mode='reflect')[:800, :800]

for iter in xrange(100):
    deg = np.random.randint(0, 360)
    theta = np.deg2rad(deg)

    tx = np.random.randint(0, 200)
    ty = np.random.randint(0, 200)

    S, C = np.sin(theta), np.cos(theta)

    # Rotation matrix, angle theta, translation tx, ty
    H = np.array([[C, -S, tx],
                  [S,  C, ty],
                  [0,  0, 1]])

    # Translation matrix to shift the image center to the origin
    r, c = img.shape
    T = np.array([[1, 0, -c / 2.],
                  [0, 1, -r / 2.],
                  [0, 0, 1]])

    # Skew, for perspective
    S = np.array([[1, 0, 0],
                  [0, 1.2, 0],
                  [0, 1e-3, 1]])

    img_rot = transform.fast_homography(img, H, mode='mirror')
    img_rot_center_skew = transform.fast_homography(img, S.dot(np.linalg.inv(T).dot(H).dot(T)), mode='mirror')
    #img_rot_center_skew = arraypad.pad(img_rot_center_skew, 512, mode='reflect')


    rf = 95
    j, i = np.random.randint(0, len(img_rot_center_skew)-rf, 2)
    print j, i
    img_rot_center_skew2 = img_rot_center_skew[j:j+rf, i:i+rf]
    img2 = img[j:j+rf, i:i+rf]
    misc.imsave('img_%0d.png' % iter, img2)
    misc.imsave('img_rot_%0d.png' % iter, img_rot_center_skew2)

    #f, (ax0, ax2) = plt.subplots(1, 2)
    #ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    #ax2.imshow(img_rot_center_skew, cmap=plt.cm.gray, interpolation='nearest')
    #plt.show()

    #img = arraypad.pad(img, 512, mode='reflect')

    #for iter in xrange(100):
        #print iter
        #j, i = np.random.randint(0, len(img_rot_center_skew)-rf, 2)
        #print j, i
        #img_rot_center_skew2 = img_rot_center_skew[j:j+rf, i:i+rf]
        #img2 = img[j:j+rf, i:i+rf]
        #misc.imsave('img_%0d.png' % iter, img2)
        #misc.imsave('img_rot_%0d.png' % iter, img_rot_center_skew2)
