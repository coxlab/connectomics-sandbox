#!/bin/bash

source ~/.bashrc

# -- name of the virtualenv directory
VENV_NAME=connectomics-sandbox-slm_experiments-system

# -- export two environment variables
export CONNECTOMICS_HP_BASE_PATH="/path/to/dataset"
export

# -- shell dependencies
command -v mkvirtualenv

# -- current working directory
CWD=`pwd`

(cd ../../ && git submodule init && git submodule update)

mkvirtualenv --system-site-packages ${VENV_NAME}

# -- "frozen" dependencies
pip install "scipy>=0.9.0"
pip install "numpy>=1.6.1"
pip install "scikits-image>=0.5"

pip install --no-deps -I git+https://github.com/npinto/asgd.git
pip install --no-deps -I git+https://github.com/npinto/mcc.git
pip install --no-deps -I git+https://github.com/npinto/bangmetric.git
pip install --no-deps -I git+https://github.com/davidcox/genson.git
pip install --no-deps -I git+https://github.com/nsf-ri-ubicv/sthor.git

# -- "active" dependencies
(cd ${CWD}/../external/coxlabdata && python setup.py develop)

echo
echo "******************************************************"
echo "Done! Don't forget to run"
echo "$ workon ${VENV_NAME}"
echo "******************************************************"
echo
