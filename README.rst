Requirements
============

working versions of

1. ``virtualenv``
2. ``virtualenvwrapper``
3. ``git``

Set up the environment
======================

1. create a ``tmp/`` directory in which you clone the connectomics
   Github repository

        $ cd tmp/
        $ git clone git@github.com:poilvert/connectomics-sandbox.git

2. move into ``connectomics-sandbox/``

        $ cd connectomics-sandbox/

3. execute ``bootstrap.sh``

        $ chmod +x bootstrap.sh

        $ ./bootstrap.sh

        $ [enter the path to the root directory for your virtualenv] ROOTDIR

4. wait for all the installation to take place.
5. remove ``tmp/``
6. Instead of performing the following commands (7, 8 and 9) you could also directly
   issue a

        $ workon connectomics-sandbox

7. move into ``$ROOTDIR/connectomics-sandbox/``
8. source the virtual environment

        $ source bin/activate

9. move into ``connectomics``

        $ cd connectomics

and start to play with the codes (see below).

Using the codes
===============

the directory contains mostly three codes:

    1. ``generate_connectome_v1_features.py`` to generate the binary file that
       that will be mem-mapped
    2. ``compute_tm.py`` that train and test a classifier given some command
       line options
    3. ``extract_and_plot_predicted_target_maps.py`` used to extract the target
       maps and store the results as png images in a directory

the first code is executed directly while the last two have command line
interfaces that are hopefully self-explanatory.
An important file to carefully update is ``parameters.py`` where some path
and options are defined. This files is used by the above codes so make sure
you have proper paths and option values.

Note
====

So far the generation code is ahead of ``compute_tm.py`` so you cannot use
the latter unless you already have computed another set of pikle files which
are not produced by ``generate_connectome_v1_features.py``.

Also, the ``bootstrap.sh`` script is very not general, so it will need some
hard coded path removals.
