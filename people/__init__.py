from __future__ import absolute_import

import sys
if sys.version_info[0] < 3:
    sys.stderr.write('You must use Python 3\n')
    sys.exit()

__version__ = '0.1.0'

info = 'Initializing people v.{}\n'.format(__version__)
sys.stderr.write(info)

def datasetpath(dataset_name):
    import socket
    hostname = socket.gethostname()

    if hostname == "turing":
        return "/home/diogo/Datasets/" + str(dataset_name)

    if hostname == "emeritus":
        return "/storage/diogo/Datasets/"+ str(dataset_name)

    if (hostname == "tectonix") or \
            (hostname == "pascalix.bim-ensea.fr") or \
            (hostname == "triplepatte") or \
            (hostname == "baba"):
        return "/local/diogluvi/" + str(dataset_name)

    if (hostname == "harenbaltix") or (hostname == "elevedelix"):
        return "/local/Datasets/" + str(dataset_name)


import os
keras_git = os.environ.get('HOME') + '/git/fchollet/keras'
info = 'Using keras'
if os.path.isdir(keras_git):
    sys.path.insert(0, keras_git)
    info += ' from "{}"'.format(keras_git)

try:
    sys.stderr.write('CUDA_VISIBLE_DEVICES: '
            + str(os.environ['CUDA_VISIBLE_DEVICES']) + '\n')
except:
    sys.stderr.write('CUDA_VISIBLE_DEVICES not defined\n')

import keras
info += ' version "{}"\n'.format(keras.__version__)

sys.stderr.write(info)

import numpy as np
try:
    if hasattr(np.core.arrayprint, '_line_width'):
        np.core.arrayprint._line_width = 159
    elif hasattr(np.core.arrayprint, '_format_options'):
        np.core.arrayprint._format_options['linewidth'] = 159
except:
    pass
