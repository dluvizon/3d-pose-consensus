import __config__

import json
import numpy as np
import random

from people import datasetpath
from people.datasets import Human36MTest
from people.model import People3D
from people.utils import *

from people.datasets.human36m import ZBOUND

pred_file = 'output/h36m_test_bboxes/h36m_test_pred_ff9bc9.json'
afmat_file = 'output/h36m_test_bboxes/h36m_test_afmat_ff9bc9.json'
out_file = 'output/h36m_test_bboxes/h36m_test_bbox_ff9bc9.json'

with open(pred_file, 'r') as fid:
    pred = np.array(json.load(fid))
with open(afmat_file, 'r') as fid:
    afmat = np.array(json.load(fid))

p2d = transform_pose_sequence(afmat, pred[:, :, :2].copy(), inverse=True)

bbox = PoseBBox(p2d, relsize=1.6, square=True)

# crop_res = (1000, 1000)
# anchors_size = [(125, 125)]
# num_predictions = 2
# image_div = 8
# poselayout = est34j3d

# h36m_dataconf = DataConfig(
        # crop_resolution=crop_res,
        # )

# h36m = Human36MTest(datasetpath('Human3.6M'), h36m_dataconf, anchors_size,
        # image_div=8, bbox_file=None, preprocess_mode='caffe')

# """Human3.6H testing samples."""
# h36m_te = BatchLoader(h36m, ['frame', 'aref'], ['afmat', 'camera'],
        # TEST_MODE, batch_size=1, shuffle=False)

# i = 0
# x, y = h36m_te[i]
# img = x[0][0, :, :, ::-1]
# plot.draw(img, p2d[i], bboxes=bbox[i], abs_pos=True)


bbox = bbox[:].tolist()

with open(out_file, 'w') as fid:
    json.dump(bbox, fid)

