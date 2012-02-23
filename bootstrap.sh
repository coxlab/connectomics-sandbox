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
cd $VENV_HOME/$PROJECT_NAME

# -- now cloning the project Github repository
git clone git@github.com:coxlab/connectomics-sandbox.git
cd $PROJECT_NAME

# -- once in the root directory we will install the
#    project package
python setup.py develop
