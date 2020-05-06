import __config__

import os
import sys

import json
import numpy as np
import people

import keras.backend as K

from people.datasets import Human36M

from people.evaluation import average_world_poses
from people.evaluation import eval_human36m_activities

from people.utils import *


from people import datasetpath
from people.datasets.human36m import ZBOUND


poselayout = est34j3d
num_joints = poselayout.num_joints
anchor_id = 0
hf = 0
crop_mode = '3d'
info = 'hf%d_bbox%s-gt' % (hf, crop_mode)
info0 = 'hf%d_bbox%s-gt' % (0, crop_mode)

# Required to get ACTION_LABELS
h36m = Human36M(datasetpath('Human3.6M'), DataConfig(), [(32, 32)])

saveddir = 'output/h36m_pred_cam_val_hf0_100_2d6db72'

p_h36m = np.load(os.path.join(saveddir, 'p_h36m_%s.npy' % info))
pw_h36m = np.load(os.path.join(saveddir, 'pw_h36m_%s.npy' % info0))
afmat_h36m = np.load(os.path.join(saveddir, 'afmat_h36m_%s.npy' % info))
rootz_h36m = np.load(os.path.join(saveddir, 'rootz_h36m_%s.npy' % info0))
scam_h36m = np.load(os.path.join(saveddir, 'scam_h36m_%s.npy' % info0))
action_h36m = np.load(os.path.join(saveddir, 'action_h36m_%s.npy' % info0))
pred_z = np.load(os.path.join(saveddir, 'pred_absz_%s.npy' % info))
pred_p = np.load(os.path.join(saveddir, 'pred_pose_%s.npy' % info))
pred_pc = np.load(os.path.join(saveddir, 'pred_pc_%s.npy' % info))
with open(os.path.join(saveddir, 'meta_%s.json' % info), 'r') as fid:
    meta_h36m = json.load(fid)

if hf == 1:
    p_h36m = p_h36m[:, :, poselayout.map_hflip, :]
    pred_p = pred_p[:, :, poselayout.map_hflip, :]

pw_h36m = pw_h36m[:, h36m23j3d.map_to_pa17j, :]
p_h36m = p_h36m[:, anchor_id, poselayout.map_to_pa17j, :3]
pred_p = pred_p[:, anchor_id, poselayout.map_to_pa17j, :3]
pred_pc = pred_pc[:, poselayout.map_to_pa17j, :3]

cameras = [camera_deserialize(c) for c in scam_h36m]
rootz = rootz_h36m[:, anchor_id:anchor_id+1]*(ZBOUND[1] - ZBOUND[0]) + ZBOUND[0]
pred_z = pred_z[:, anchor_id:anchor_id+1]*(ZBOUND[1] - ZBOUND[0]) + ZBOUND[0]


"""Sanity check."""
pose_w = inverse_project_pose_to_camera_ref(p_h36m, rootz, afmat_h36m,
        cameras, resol_z=2000., project_to_world=True)
err_w = np.mean(np.sqrt(np.sum(
                np.square(pw_h36m - pose_w), axis=-1)))
print ('Err_e on GT poses projected to world: ', err_w)

pose_c1 = inverse_project_pose_to_camera_ref(p_h36m, rootz, afmat_h36m,
        cameras, resol_z=2000., project_to_world=False)

pose_c_gt = np.nan * np.ones(pw_h36m.shape)
for i in range(len(pose_c_gt)):
    pose_c_gt[i] = project_world2camera(cameras[i], pw_h36m[i])

err_c = np.mean(np.sqrt(np.sum(
                np.square(pose_c1 - pose_c_gt), axis=-1)))
print ('Err_e on GT poses projected to camera: ', err_c)


"""Convert predictions to world coordinates."""
# for camidx in [[0], [1,2], [1, 4], [1, 2, 3], [1, 2, 3, 4]]:
# for camidx in [[0]]:
for camidx in [[1, 2], [1, 2, 3, 4]]:
    # pred_w = inverse_project_pose_to_camera_ref(pred_p.copy(), pred_z.copy(),
            # afmat_h36m, cameras, resol_z=2000., project_to_world=True)
    pred_w = np.nan * np.ones(pred_pc.shape)
    for j in range(len(pred_pc)):
        cam = cameras[j]
        pred_w[j] = (np.matmul(cam.R_inv, pred_pc[j].T) + cam.t).T


    pred_w2 = average_world_poses(pred_w.copy(), meta_h36m, camera_indexes=camidx)

    pose_w2c = np.nan * np.ones(pred_w.shape)
    for i in range(len(pose_w2c)):
        pose_w2c[i] = project_world2camera(cameras[i], pred_w2[i])

    err_pred = np.mean(np.sqrt(np.sum(
                    np.square(pose_w2c - pose_c_gt), axis=-1)))
    print ('ABS Err_e on predicted poses using averaged cameras ' \
            + str(camidx) + ' : ' + str(err_pred) + ' mm')

    print ('ERROR ON ABSOLUTE MULTICAM')
    eval_human36m_activities(pose_c_gt, pose_w2c, action_h36m)

    pose_w2c -= pose_w2c[:, 0:1, :]
    pose_c_gt_rel = pose_c_gt - pose_c_gt[:, 0:1, :]

    print ('ERROR ON RELATIVE MULTICAM')
    eval_human36m_activities(pose_c_gt_rel, pose_w2c, action_h36m)


    err_pred = np.mean(np.sqrt(np.sum(
                    np.square(pose_w2c - pose_c_gt_rel), axis=-1)))
    print ('REL Err_e on predicted poses using averaged cameras ' \
            + str(camidx) + ' : ' + str(err_pred) + ' mm')

exit()

"""Evaluate single-view with horizontal flip."""
afmat0_h36m = np.load(os.path.join(saveddir, 'afmat_h36m_hf0.npy'))
afmat1_h36m = np.load(os.path.join(saveddir, 'afmat_h36m_hf1.npy'))
pred_z0 = np.load(os.path.join(saveddir, 'pred_absz_hf0.npy'))
pred_z1 = np.load(os.path.join(saveddir, 'pred_absz_hf1.npy'))
pred_p0 = np.load(os.path.join(saveddir, 'pred_pose_hf0.npy'))
pred_p1 = np.load(os.path.join(saveddir, 'pred_pose_hf1.npy'))

pred_p1 = pred_p1[:, :, poselayout.map_hflip, :]

pred_p0 = pred_p0[:, anchor_id, poselayout.map_to_pa17j, :3]
pred_p1 = pred_p1[:, anchor_id, poselayout.map_to_pa17j, :3]

pred_z0 = pred_z0[:, anchor_id:anchor_id+1]*(ZBOUND[1] - ZBOUND[0]) + ZBOUND[0]
pred_z1 = pred_z1[:, anchor_id:anchor_id+1]*(ZBOUND[1] - ZBOUND[0]) + ZBOUND[0]

pred_w0 = inverse_project_pose_to_camera_ref(pred_p0.copy(), pred_z0.copy(),
        afmat0_h36m, cameras, resol_z=2000., project_to_world=True)
pred_w1 = inverse_project_pose_to_camera_ref(pred_p1.copy(), pred_z1.copy(),
        afmat1_h36m, cameras, resol_z=2000., project_to_world=True)


err = np.mean(np.sqrt(np.sum(np.square(pred_w0 - pred_w1), axis=-1)))
print ('Abs error hflip vs single camera', err)

aa = pred_w0 - pred_w0[:, :1, :]
bb = pred_w1 - pred_w1[:, :1, :]
err = np.mean(np.sqrt(np.sum(np.square(aa - bb), axis=-1)))
print ('Rel error hflip vs single camera', err)

pred_w = (pred_w0 + pred_w1) / 2.

pose_w2c = np.nan * np.ones(pred_w.shape)
for i in range(len(pose_w2c)):
    pose_w2c[i] = project_world2camera(cameras[i], pred_w[i])

err_pred = np.mean(np.sqrt(np.sum(
                np.square(pose_w2c - pose_c_gt), axis=-1)))
print ('ABS Err_e on predicted poses single view h. flip ' \
        + str(err_pred) + ' mm')

print ('ERROR ON ABSOLUTE SINGLE')
eval_human36m_activities(pose_c_gt, pose_w2c, action_h36m)


pose_w2c -= pose_w2c[:, 0:1, :]
pose_c_gt_rel = pose_c_gt - pose_c_gt[:, 0:1, :]

print ('ERROR ON RELATIVE SINGLE')
eval_human36m_activities(pose_c_gt_rel, pose_w2c, action_h36m)

err_pred = np.mean(np.sqrt(np.sum(
                np.square(pose_w2c - pose_c_gt_rel), axis=-1)))
print ('REL Err_e on predicted poses single view h. flip ' \
        + str(err_pred) + ' mm')

