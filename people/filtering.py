import numpy as np

def compute_pose_dissimilarity(poses, pose_ref, joint_ratio=0.3):
    p = np.tile(np.expand_dims(pose_ref, axis=0), (len(poses), 1, 1))
    diff = poses[:,:,:-1] - p[:,:,:-1]
    diss = p[:, :, -1] * poses[:, :, -1] \
            * np.sqrt(np.nansum(np.square(diff), axis=-1))
    diss.sort(axis=-1)
    min_num_joints = int(joint_ratio * diss.shape[-1])

    return np.nanmean(diss[:, :min_num_joints], axis=-1)

def average_poses(poses):
    c = np.tile(np.square(poses[:, :, -1:]), (1, 1, poses.shape[-1]))
    c = np.clip(c, 1e-4, None)
    p = np.nansum(c * poses, axis=0, keepdims=True) \
              / np.nansum(c, axis=0, keepdims=True)

    return p

def multipose_nms(poses, hardmax=False, min_avg_conf=0.3, remove_diss_coef=1.,
        max_people=10):

    """Remove some poses with low average confidence."""
    meanc = np.nanmean(poses[:,:,-1], axis=-1)
    poses = poses[meanc > min_avg_conf].copy()

    meanc = np.nanmean(poses[:,:,-1], axis=-1)
    poses = poses[np.argsort(meanc)[::-1]]

    output = []
    while len(poses) > 0:
        stdref = np.nanstd(poses[:,:,:-1])
        softmax_diss = stdref / 5.
        min_remove_diss = remove_diss_coef * stdref

        diss = compute_pose_dissimilarity(poses, poses[0])
        print (diss)

        if hardmax:
            output.append(poses[:1])
        else:
            output.append(average_poses(poses[diss < softmax_diss]))

        if len(output) >= max_people:
            break

        poses = poses[diss > min_remove_diss]

    return np.concatenate(output, axis=0)

