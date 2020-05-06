import __config__

import os
import sys

import numpy as np
import people

from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
import keras.backend as K

from people import datasetpath
from people.datasets import MpiInf3D
from people.datasets import MPII
from people.model import People3D
from people.losses import structural_multiperson_loss_builder
from people.losses import absz_multiperson_loss
from people.callbacks import SaveModel
from people.callbacks import H36MEvalCallback
from people.callbacks import MpiiEvalCallback
from people.utils import *

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

start_lr = 0.001
crop_res = (256, 256)
anchors_size = [(32, 32), (24, 32), (32, 24), (15, 20), (20, 15)]
num_predictions = 4
poselayout = pa17j3d
num_joints = poselayout.num_joints
batch_size_mpii = 12
batch_size_h36m = 12
weights_path = os.path.join(logdir, 'weights_h36m_{epoch:03d}.hdf5')

h36m_dataconf = DataConfig(
        crop_resolution=crop_res,
        angles=np.array(range(-10, 10+1)),
        scales=np.array([.7, 1.0, 1.3]),
        trans_x=np.array(range(-40, 40+1)),
        trans_y=np.array(range(-10, 10+1)),
        geoocclusion=np.array(range(10, 90)),
        )

mpii_dataconf = DataConfig(
        crop_resolution=crop_res,
        angles=np.array(range(-30, 30+1)),
        scales=np.array([0.7, 1.0, 1.3]),
        )

model = People3D(crop_res + (3,), anchors_size, num_joints,
        num_predictions=num_predictions)

mpii = MPII(datasetpath('MPII'), mpii_dataconf, anchors_size,
        image_div=8,
        poselayout=poselayout,
        preprocess_mode='caffe')

h36m = Human36M(datasetpath('Human3.6M'), h36m_dataconf, anchors_size,
        image_div=8,
        poselayout=poselayout,
        preprocess_mode='caffe')

"""Training mixed data."""
data_tr = BatchLoader([h36m, mpii], ['frame', 'aref'], ['rootz', 'pose'],
        TRAIN_MODE, batch_size=[batch_size_h36m, batch_size_mpii],
        num_predictions=[1, num_predictions], shuffle=True)

"""Human3.6H validation samples."""
h36m_val = BatchLoader(h36m, ['frame', 'aref'],
        ['pose_w', 'afmat', 'rootz', 'camera', 'action'],
        VALID_MODE, batch_size=h36m.get_length(VALID_MODE), shuffle=True)
printcn('Pre-loading Human3.6M validation data...', OKBLUE)
[x_h36m, aref_h36m], [pw_h36m, afmat_h36m, rootz_h36m, scam_h36m, action_h36m] \
        = h36m_val[0]

"""MPII validation samples."""
mpii_val = BatchLoader(mpii, ['frame', 'aref'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
printcn('Pre-loading MPII validation data...', OKBLUE)
[x_mpii, aref_mpii], [p_mpii, afmat_mpii, head_mpii] = mpii_val[0]

"""Define the loss functions and compile the model."""
struct_loss = structural_multiperson_loss_builder()
losses = [absz_multiperson_loss] + num_predictions * [struct_loss]
model.compile(loss=losses, optimizer=RMSprop(lr=start_lr))
model.summary(line_length=159)

def lr_scheduler(epoch, lr):

    if epoch in [70, 90]:
        newlr = 0.2*lr
        printcn('lr_scheduler: lr %g -> %g @ %d' % (lr, newlr, epoch), WARNING)
    else:
        newlr = lr
        printcn('lr_scheduler: lr %g @ %d' % (newlr, epoch), OKBLUE)

    return newlr

callbacks = []
callbacks.append(LearningRateScheduler(lr_scheduler))
callbacks.append(SaveModel(weights_path))

callback_h36m = H36MEvalCallback(x_h36m, aref_h36m, pw_h36m, afmat_h36m,
        scam_h36m, action_h36m, rootz=rootz_h36m,
        logdir=logdir)
callbacks.append(callback_h36m)

callback_mpii = MpiiEvalCallback(x_mpii, aref_mpii, p_mpii, afmat_mpii,
        head_mpii, map_to_mpii=poselayout.map_to_mpii, logdir=logdir)
callbacks.append(callback_mpii)

steps_per_epoch = int(mpii.get_length(TRAIN_MODE) / batch_size_mpii)

model.fit_generator(data_tr,
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        callbacks=callbacks,
        workers=8,
        initial_epoch=0)

