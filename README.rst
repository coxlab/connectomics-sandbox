Installation Steps
==================

1. Create a Python virtual environment on your machine (this will allows you to install
   packages without interfering with your *system* Python environment)

    $ mkvirtualenv --system-site-packages --distribute connectomics_box

2. Git clone the connectomics repository

    $ git clone <connectomics-box_repository>.git

3. Create a **dependencies** directory

    $ mkdir dependencies && cd dependencies

4. Install approriate external Python modules

    $ git clone git://github.com/npinto/bangmetric.git
    $ cd bangmetric
    $ python setup.py develop
    $ cd ..
    $ git clone git@github.com:poilvert/bangreadout.git
    $ cd bangreadout
    $ git checkout squarehingeclf
    $ python setup.py develop
    $ git clone git@github.com:nsf-ri-ubicv/sthor.git
    $ cd sthor
    $ git checkout isbi12
    $ python setup.py develop

5. Export the Fiji path

    $ export FIJI_EXE_PATH=/path/to/fiji-linux64

6. Go into the **experiments** directory of the connectomics box repository, and let the system
   know where the ISBI train/test PNG images are by editing the file ``get_isbi_images.py``

    $ cd /path/to/connectomics/experiments
    $ vim get_isbi_images.py

   and put the path in the variable *base_path*

Testing your installation
=========================

Test ! In ``driver.py`` set the training image list and testing image list to :

    DEFAULT_TRN_IDX = [0, 1, 2]
    DEFAULT_TST_IDX = [29]
    DEFAULT_SAVE = False
    DEFAULT_ROTATE = False
    DEFAULT_USE_TRUE_TST_IMG = False

In the main source code ``models.py``, make sure to select the custom cascade dictionnary. For
this, uncomment the line :

    info_dict = custom_cascade

at the beginning of the *process* function. Then execute the code as follows :

    $ python driver.py models.process

the metric values should be close to something like :

    mean Average Precision:  0.987
    mean Pearson coef.    :  0.721
    mean Pixel Error      :  0.058
    mean Rand Error       :  0.369
    mean Warping Error    :  0.003

Why a driver ?
==============

Everything goes through the ``driver.py`` program. The idea is that instead
of worrying about possible cross validation folds, metrics computation and storage
of data one can use the driver directly and focus on writing code for better models
and classifiers.

How to test the driver ?
========================

    $ python driver.py -h

and a menu displaying the different positional and non-positional arguments
will be printed on screen. For a more complete test, follow the intructions in the section
*Testing your installation*.

If you want to use MongoDB to store the data
============================================

The driver dumps all the data in a MongoDB database by default. In order to
make it work, edit the appropriate default parameters related to MongoDB in
``driver.py``.

If you'd like to store your data into a Pickle file instead, just type
``--no_mongo_store`` on the command-line when using the driver.

How to use the driver with your program ?
=========================================

The driver is a front-end program that will take a python module of yours
(e.g. ``mymodule.py``) which contains a certain function *myfunction*
(that is responsible for the processing of the connectomics images and for
producing boundary detection maps), and use that function internally.

The standard command-line is:

    $ python driver.py mymodule.myfunction myfunction_args <driver_args>

where ``mymodule`` is the path to your python module (e.g. ``mydir/mymodule``),
*myfunction* is the name of the function in your module to use for the
computation. *myfunction_args* are all the extra args to pass to your function.
Finally all other non-positional arguments of the driver follow.

The driver performs the following steps:

1. It first uses a program to extract the training and testing images and organize
   them as a list of lists. Each list representing a cross-validation fold.
   In the code, that list of lists is called ``tasks``.

2. Then your function is called by the driver. The goal of your function is to take
   the ``tasks`` and train a model to finally produce some predictions on *test*
   images.

3. Finally the driver *collects* the predictions from the model and computes a set
   of metrics if available (this is only the case if there exists ground truths
   for the testing images).

The line of code in the driver that calls your function is :

    output_true, output_pred, to_save = function(tasks, args.function_arguments)

This line gives a set of constraints that your function must satisfy. First, the
function must take the ``tasks`` as first argument (plus potentially as many as
one wants as extra arguments). Then it must return three outputs. The first two
of which must be 4D tensors of shape *[ni, h, w, nf]*. Possibly, if no ground
truth images were present for the testing images, ``output_true`` should be an
empty list of array. The last output can be anything that the user would like
to store in a Pickle or a MongoDB database.
