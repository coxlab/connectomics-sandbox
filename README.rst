Requirements
============

General-purpose packages
------------------------

As in most Python-based codes, you will need to have working versions of ::

    numpy
    scipy
    numexpr
    pip

Reading images and Machine Learning
-----------------------------------

For reading images, and doing some machine learning, you will need to install ::

    scikits-image
    theano
    scikits-learn

``scikits-image`` will also need to have the ``freeimage`` plugin installed. On Gentoo
Linux, the following command should work ::

    emerge media-libs/freeimage

On Ubuntu Linux, something like this instead should work ::

    sudo apt-get install libfreeimage3

Database-related packages
-------------------------

Finally, in order to be able to dump data in a MongoDB database, you will need ::

    pymongo

If you dump your data onto another computer, you don't need to have MongoDB installed
on your machine, as long as it is installed on the "server" where you dump your data
with ``pymongo``.

Computing metrics
-----------------

In order to compute the metrics associated with the ISBI challenge, you need to install a
recent version of Fiji.

1. Go to http://fiji.sc/wiki/index.php/Downloads

2. Once downloaded, decompress the archive

3. Start Fiji with the executable provided, and update Fiji

I would advise to download and install ``Fiji`` on your own machine with a Graphical User
Interface. Because the "update" to Fiji need a GUI. Once you update Fiji, you can then do
a safe copy onto the machine on which you will perform the calculations. This is quite
useful in particular if you need a working version of ``Fiji`` onto a cluster that generally
does not have a graphical interface.

Installation Steps
==================

1. Create a Python virtual environment on your machine (this will allows you to install
   packages without interfering with your *system* Python environment) ::

    $ mkvirtualenv --system-site-packages --distribute connectomics_box

2. Git clone the connectomics repository ::

    $ git clone <connectomics-box_repository>.git

3. Create a **dependencies** directory ::

    $ mkdir dependencies && cd dependencies

4. Install approriate external Python modules ::

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

5. Export the Fiji path ::

    $ export FIJI_EXE_PATH=/path/to/fiji-linux64

6. Go into the **experiments** directory of the connectomics box repository, and let the system
   know where the ISBI train/test PNG images are by editing the file ``get_isbi_images.py`` ::

    $ cd /path/to/connectomics/experiments
    $ vim get_isbi_images.py

   and put the path in the variable *base_path*. Equivalently, you can define an environment
   variable ``ISBI_PATH`` and exporting the proper path into that variable.

Testing your installation
=========================

Test ! In ``driver.py`` set the training image list and testing image list to ::

    DEFAULT_TRN_IDX = [0, 1, 2]
    DEFAULT_TST_IDX = [29]
    DEFAULT_SAVE = False
    DEFAULT_ROTATE = False
    DEFAULT_USE_TRUE_TST_IMG = False

In the main source code ``models.py``, make sure to select the custom cascade dictionnary. For
this, uncomment the line ::

    info_dict = custom_cascade

at the beginning of the *process* function. Then execute the code as follows ::

    $ python driver.py models.process

the metric values should be something around these values ::

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

::

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

The standard command to use the driver with your custom code is ::

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

The line of code in the driver that calls your function is ::

    output_true, output_pred, to_save = function(tasks, args.function_arguments)

This line gives a set of constraints that your function must satisfy. First, the
function must take the ``tasks`` as first argument (plus potentially as many as
one wants as extra arguments). Then it must return three outputs. The first two
of which must be 4D tensors of shape *[ni, h, w, nf]*. Possibly, if no ground
truth images were present for the testing images, ``output_true`` should be an
empty list or array. The last output can be anything that the user would like
to store in a Pickle or a MongoDB database.

Some tips on using the codes on computing clusters
==================================================

When using the code on a cluster of computers, it happens that ``theano`` wants
to dump all its compiled sources into a **cache** directory, which by default, is
set to be in your ``home`` directory. The problem is that when many nodes of the
cluster want to compile their theano expressions, they all share that common
cache directory, and this causes a lot of reading/writing which slows down
tremendously the system, especially since your ``home`` folder will be shared by
the NFS system of the cluster.

In order to avoid all this, clusters generally provide *local scratch* directories
on each node. These directories are *local* to the node, which means that they are
not on the NFS system. The line that calls the driver to perform the calculation
you want should then look something like this ::

    THEANO_FLAGS='base_compiledir=/path/to/local/scratch' python driver.py <rest of command here>
