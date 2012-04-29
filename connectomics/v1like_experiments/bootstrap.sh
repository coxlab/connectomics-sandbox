#!/bin/bash

source ~/.bashrc

# -- shell dependencies
command -v mkvirtualenv

CWD=`pwd`  # current working dir

(cd ../../ && git submodule init && git submodule update)

mkvirtualenv --system-site-packages connectomics-sandbox-v1like_experiments-system

# -- "frozen" dependencies
pip install --no-deps -I git+https://github.com/npinto/asgd.git
pip install --no-deps -I git+https://github.com/npinto/bangmetric.git

# -- "active" dependencies
(cd ${CWD}/../external/coxlabdata && python setup.py develop)
