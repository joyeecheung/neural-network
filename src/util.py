#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import os

ASSET_DIR = 'asset'
TRAIN_FNAME = 'digitstra.txt'
TEST_FNAME = 'digitstest.txt'
CURVE = 'learning-curve.png'
ERROR = 'error-curve.png'


def get_filenames():
    """Return a named tuple of filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    asset_path = os.path.join(parent_dir, ASSET_DIR)
    filenames = map(lambda name: os.path.join(asset_path, name),
                    (TRAIN_FNAME, TEST_FNAME, CURVE, ERROR))

    fn = namedtuple('Files', ['train', 'test', 'curve', 'error'])
    return fn(*filenames)
