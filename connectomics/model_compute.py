#!/usr/bin/env python

"""
This program will:
    1. create a cross validated screening task with
       train/test and validation sets
    2. trains a classifier for every cross validation
       fold and computes the performance for both the
       testing images and validation images
"""

from trn_tst_val_generator import generate as generate_cv
