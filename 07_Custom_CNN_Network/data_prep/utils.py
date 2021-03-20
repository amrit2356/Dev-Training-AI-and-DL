"""Utils"""

import os
import shutil

import numpy as np


def create_path(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)
