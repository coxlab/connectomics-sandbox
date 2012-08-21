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
   know where the ISBI train/test PNG images are by editing the file *get_isbi_images.py*

    $ cd /path/to/connectomics/experiments
    $ vim get_isbi_images.py

   and put the path in the variable *base_path*

7. Test ! In *driver.py* set the training image list and testing image list to :

    DEFAULT_TRN_IDX = [0, 1, 2]
    DEFAULT_TST_IDX = [29]
    DEFAULT_SAVE = False
    DEFAULT_ROTATE = False
    DEFAULT_USE_TRUE_TST_IMG = False

   In the main source code *models.py*, make sure to select the custom cascade dictionnary. For
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
