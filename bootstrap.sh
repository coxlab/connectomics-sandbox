#!/bin/bash

# ----------------------------------------------------
#  bootstrap script that will set up a Python virtual
#  environment with all the required dependencies and
#  make sure that the codes can be run smoothly
# ----------------------------------------------------

# -- name of the project root directory
export PROJECT_NAME='connectomics-sandbox'

# -- where to create the virtual environment
echo 'path to virtualenv home directory'
read VENV_HOME

# -- create the virtual environment
export WORKON_HOME=$VENV_HOME
source /usr/bin/virtualenvwrapper.sh
mkvirtualenv $PROJECT_NAME

# -- move into the project root directory
cd $WORKON_HOME/$PROJECT_NAME
source bin/activate

# -- install some needed packages
pip install -I cython

# -- now cloning the project Github repository
git clone git@github.com:poilvert/connectomics-sandbox.git
cd $PROJECT_NAME

# -- git submodules activation
git submodule init
git submodule update

# -- installing the submodules in develop mode
for dir in "connectomics"/"external"/*
do
    echo "installing $dir in develop mode"
    cd $dir
    python setup.py develop
    cd ../../..
done

# -- install project package in develop mode
python setup.py develop
