Requirements
============

working versions of

1. ``virtualenv``
2. ``virtualenvwrapper``
3. git

Set up the environment
======================

1. create a ``tmp/`` directory in which you clone the connectomics
   Github repository

        $ git clone git@github.com:coxlab/connectomics-sandbox.git

2. move into ``connectomics-sandbox/``
3. execute ``bootstrap.sh``

        $ . ./bootstrap.sh

This script will create a Python virtual environment on your machine
(using virtualenvwrapper), move into the virtual environment root
directory, git clone the repo (again! I know) and run the ``setup.py``
script from there.

After these steps you can remove the ``tmp/`` directory you just created
and type

    $ workon connectomics-sandbox

