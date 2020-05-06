import numpy as np

def pose_dissimilarity(poses):
    pass

def meanshift_clustering(poses):
    """
    # Arguments
        poses: Array of shape (num_poses, num_joints, dim+1)

    # Return
        Array with resulting `k` poses with shape (k, num_joints, dim+1).
    """

def mean_confident_pose(poses, absz=None):
    """
    # Arguments
        poses: Array with shape (num_poses, num_joints, dim+1)
        absz: Array with shape (num_poses,)

    # Return
        Array with the centroid pose with shape (num_joints, dim+1)
        Absolute Z (float)
    """
    p = poses[:, :, :3]
    c = np.square(poses[:, :, 3:])
    cc = np.tile(c, (1, 1, 3))

    cp = np.sum(cc * p, axis=0) / np.clip(np.sum(cc, axis=0), 1e-7, None)
    cc = np.mean(cc[..., :1], axis=0)

    cz = np.mean(c[:, 0:2, 0], axis=-1)
    # z = np.sum(cz * absz, axis=0) / np.clip(np.sum(cz, axis=0), 1e-7, None)

    return np.concatenate([cp, cc], axis=-1)

