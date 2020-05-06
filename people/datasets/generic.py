import os

import numpy as np

from ..utils import *

def project_gt_poses_to_anchors(poses, anchors):
    """Project normalized poses to the normalized anchors coordinates.
    If multiple poses are given, use the closest pose for each anchor.

    # Parameters
        poses: single pose (num_joints, dim+1), which will be replicated for
            every anchor, one pose per anchor (num_anchors, num_joints, dim+1).
        anchors: reference anchors array (num_anchors, 4).

    # Return
        Projected poses to anchors (num_anchors, num_joints, dim+1).
    """
    assert poses.ndim in [2, 3] and anchors.ndim == 2, \
            'Invalid dimensions for poses {} and/or anchors {}'.format(
                    poses.shape, anchors.shape)

    num_anchors = anchors.shape[0]
    dim = poses.shape[-1] - 1

    if poses.ndim == 3:
        assert poses.shape[0] == num_anchors, \
                'Incompatible num_anchors on pose {}'.format(poses.shape[0])
        num_joints = poses.shape[1]
    else:
        num_joints = poses.shape[0]
        poses = np.expand_dims(poses, axis=0)
        poses = np.tile(poses, (num_anchors, 1, 1))

    poses = np.reshape(poses, (num_anchors * num_joints, -1))

    anchors = np.expand_dims(anchors, axis=1)
    anchors = np.tile(anchors, (1, num_joints, 1))
    anchors = np.reshape(anchors, (num_anchors * num_joints, -1))

    poses[:, 0:2] = (poses[:, 0:2] - anchors[:, 0:2]) \
            / (anchors[:, 2:4] - anchors[:, 0:2])

    vis = get_visible_joints(poses[:, :2])
    poses[:, dim] = np.where(np.isnan(vis), poses[:, dim], vis)

    poses = np.reshape(poses, (num_anchors, num_joints, -1))

    return poses


def inverse_project_2dposes_from_anchors(poses, anchors):
    """Project an array of normalized 2D poses on the anchors coordinates to
    the image crop normalized coordinates.

    # Parameters
        poses: poses array (num_anchors, num_joints, dim+1)
        anchors: reference anchors array (num_anchors, 4)

    # Return
        Inverse projected poses (num_anchors, num_joints, dim+1)
    """
    assert poses.ndim == 3 and anchors.ndim == 2 and anchors.shape[-1] == 4, \
            'Invalid dimensions for pose {} and/or anchors {}'.format(
                    poses.shape, anchors.shape)

    num_anchors, num_joints = poses.shape[0:2]
    poses = np.reshape(poses, (num_anchors * num_joints, -1))

    anchors = np.expand_dims(anchors, axis=1)
    anchors = np.tile(anchors, (1, num_joints, 1))
    anchors = np.reshape(anchors, (num_anchors * num_joints, -1))

    poses[:, 0:2] = poses[:, 0:2]*(anchors[:, 2:4] - anchors[:, 0:2]) \
            + anchors[:, 0:2]

    poses = np.reshape(poses, (num_anchors, num_joints, -1))

    return poses


def compute_anchors_reference(anchors, afmat, imsize):
    """Compute the anchor references (`aref` field), based on anchors, afmat,
    and ont the absolute image size (img_w, img_h).
    """
    aux = np.zeros((len(anchors), 2, 2))
    aux[:, 0, :] = 0. # corners (0, 0)
    aux[:, 1, :] = 1. # corners (1, 1)
    aux = inverse_project_2dposes_from_anchors(aux, anchors)
    aux = transform_pose_sequence(afmat, aux, inverse=True)
    xc = np.mean(aux[:, :, 0], axis=-1, keepdims=True)
    yc = np.mean(aux[:, :, 1], axis=-1, keepdims=True)
    wchc = np.abs(aux[:, 1, :] - aux[:, 0, :])
    aref = np.concatenate([xc, yc, wchc], axis=-1)
    aref[:, 0::2] /= imsize[0]
    aref[:, 1::2] /= imsize[1]

    return aref


def compute_window_reference(afmat, imsize):
    aux = np.zeros((2, 2))
    aux[0, :] = 0. # corners (0, 0)
    aux[1, :] = 1. # corners (1, 1)
    aux = transform_2d_points(afmat, aux, transpose=True, inverse=True)
    xc = np.mean(aux[:, 0], axis=-1, keepdims=True)
    yc = np.mean(aux[:, 1], axis=-1, keepdims=True)
    wchc = np.abs(aux[1, :] - aux[0, :])
    aref = np.concatenate([xc, yc, wchc], axis=-1)
    aref[0::2] /= imsize[0]
    aref[1::2] /= imsize[1]

    return aref


class GenericDataset(object):
    """Generic implementation for a dataset class.
    """

    def __init__(self,
            dataset_path,
            dataconf,
            poselayout,
            remove_outer_joints,
            preprocess_mode):

        self.dataset_path = dataset_path
        self.dataconf = dataconf
        self.poselayout = poselayout
        self.remove_outer_joints = remove_outer_joints
        self.preprocess_mode = preprocess_mode

