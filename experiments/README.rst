Syntax for the driver
=====================

The general command to use the driver is the following ::

    $ python driver.py mymodule.myfunction myfunction_args <driver options>

An example use would be ::

    $ python driver.py models.process

Using Hans Pfister dataset instead of ISBI
==========================================

For this, just add ``hp_dataset`` as a driver option ::

    $ python driver.py models.process --hp_dataset

Saving data into a pickle
=========================

Use the ``--save`` and ``--no_mongo_store`` options ::

    $ python driver.py models.process --save --no_mongo_store

Using a MongoDB database instead
================================

Simply just ``--save`` as a driver option, and don't forget to edit the
MongoDB-related default options in ``driver.py`` ::

    $ python driver.py models.process --save

For all other options
=====================

see ::

    $ python driver.py -h

or ::

    $ python driver.py --help

How to generate a TIFF file for submission
==========================================

The managing of TIFF files is still not quite smooth with Python. Here is the very few
steps you need to follow in order to save your predictions in a format suitable for a
submission to the ISBI challenge.

1. copy the file ``tifffile.py`` from the Connectome directory in the Dropbox (more
   precisely in ``Connectome/SLM_experiments/Latest_ISBI_Results/``) into your current
   directory where you have access to your predictions (e.g. the pickle file saved by
   the driver).
2. In a python console or ipython console (just type ``python`` or ``ipython``), import
   ``imsave`` from ``tifffile.py`` by typing ::

        >>> from tifffile import imsave

3. Then take the numpy array of predictions ::

        >>> import cPickle
        >>> data = cPickle.load(open('my_data.pkl', 'r'))
        >>> pred = data['output_pred']

   and make sure that the dimensions are : ``[n_images, height, width]``. Reshape or squeeze
   the array if you have to. For example, typically, the predictions will be of the following
   shape ::

        >>> pred.shape
        (30, 512, 512, 1)

   then ``squeeze`` the array like ::

        >>> pred.squeeze()
        >>> pred.shape
        (30, 512, 512)

   and finally save the 3D array as a TIFF file with ``imsave`` ::

        >>> imsave('name_of_tiff_file_you_want.tif', pred)

4. The final step consists in opening the TIFF file you just created with ``Fiji`` (the same
   program that was used to compute the performance metrics).
   Just launch Fiji with something like ::

        $ /path/to/fiji-linux64

   Then click on "File > Open" and open the TIFF file with Fiji. Once the file has been
   opened just click on "Process > Enhance Contrast", and choose the options "normalize"
   and "normalize all 30 images". Then save the new TIFF file and you should be ready to
   submit to the ISBI Challenge !
