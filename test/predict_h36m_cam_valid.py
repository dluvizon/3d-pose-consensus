import __config__

import os
import sys

import json
import numpy as np
import people

import keras.backend as K

from people import datasetpath
from people.datasets import Human36M
from people.datasets.human36m import ZBOUND
from people.datasets.human36m import BBOX_REF
from people.model import People3D
from people.model import build_pose_cam_model
from people.utils import *

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

start_lr = 0.001
crop_res = (256, 256)
anchors_size = [(32, 32)]
num_predictions = 2
image_div = 8
poselayout = est34j3d
num_joints = poselayout.num_joints
hf = 0
crop_mode = '3d'
if crop_mode is not None:
    info = 'hf%d_bbox%s-gt' % (hf, crop_mode)
else:
    info = 'hf%d_ff' % hf

h36m_dataconf = DataConfig(
        crop_resolution=crop_res,
        fixed_hflip=hf,
        )

model = People3D(crop_res + (3,), anchors_size, num_joints,
        image_div=image_div,
        growth=128,
        num_levels=4,
        num_predictions=num_predictions,
        output_vfeat=True)

model = build_pose_cam_model(model, num_predictions, ZBOUND, BBOX_REF)

model.load_weights('output/h36m_mpii_cam_100_2d6db72/weights_h36m_100.hdf5')

h36m = Human36M(datasetpath('Human3.6M'), h36m_dataconf, anchors_size,
        image_div=image_div,
        poselayout=poselayout,
        bbox_crop_mode=crop_mode,
        preprocess_mode='caffe')

"""Human3.6H validation samples."""
h36m_val = BatchLoader(h36m, ['frame', 'aref'],
        ['pose', 'pose_w', 'afmat', 'rootz', 'camera', 'action'],
        VALID_MODE, batch_size=h36m.get_length(VALID_MODE), shuffle=False)
printcn('Pre-loading Human3.6M validation data...', OKBLUE)
[x_h36m, aref_h36m], \
        [p_h36m, pw_h36m, afmat_h36m, rootz_h36m, scam_h36m, action_h36m] \
        = h36m_val[0]

meta_h36m = []
for i in range(len(x_h36m)):
    meta_h36m.append(h36m.get_meta(i, VALID_MODE))

"""Save all predictions and ground truth informations."""
pred = model.predict([x_h36m, aref_h36m], batch_size=1, verbose=1)
np.save(os.path.join(logdir, 'aref_h36m_%s.npy' % info), aref_h36m)
np.save(os.path.join(logdir, 'p_h36m_%s.npy' % info), p_h36m)
np.save(os.path.join(logdir, 'pw_h36m_%s.npy' % info), pw_h36m)
np.save(os.path.join(logdir, 'afmat_h36m_%s.npy' % info), afmat_h36m)
np.save(os.path.join(logdir, 'rootz_h36m_%s.npy' % info), rootz_h36m)
np.save(os.path.join(logdir, 'scam_h36m_%s.npy' % info), scam_h36m)
np.save(os.path.join(logdir, 'action_h36m_%s.npy' % info), action_h36m)
np.save(os.path.join(logdir, 'pred_absz_%s.npy' % info), pred[0])
np.save(os.path.join(logdir, 'pred_pose_%s.npy' % info), pred[1])
np.save(os.path.join(logdir, 'pred_pc_%s.npy' % info), pred[2])
with open(os.path.join(logdir, 'meta_%s.json' % info), 'w') as fid:
    json.dump(meta_h36m, fid)


