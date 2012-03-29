#!/bin/bash
source ~/.bashrc

VENV_NAME=connectomics-sandbox-slm_experiments-system

# -- shell dependencies
command -v mkvirtualenv

CWD=`pwd`  # current working dir

(cd ../../ && git submodule init && git submodule update)

mkvirtualenv --system-site-packages ${VENV_NAME}

# -- "frozen" dependencies
pip install "scipy>=0.10.0"
pip install "numpy>=1.6.1"
pip install "scikits-image>=0.5"

pip install --no-deps -I git+https://github.com/npinto/asgd.git
pip install --no-deps -I git+https://github.com/npinto/mcc.git
pip install --no-deps -I git+https://github.com/npinto/bangmetric.git
pip install --no-deps -I git+https://github.com/davidcox/genson.git

# -- "active" dependencies
(cd ${CWD}/../external/coxlabdata && python setup.py develop)
(cd ${CWD}/../external/sthor && git remote add poilvert git@github.com:poilvert/sthor.git && git checkout slmnew && python setup.py develop)

echo
echo "******************************************************"
echo "Done! Don't forget to run"
echo "$ workon ${VENV_NAME}"
echo "******************************************************"
echo
