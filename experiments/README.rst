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
