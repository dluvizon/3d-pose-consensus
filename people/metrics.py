import numpy as np

from .utils import *


def _norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2), axis=axis))


def _valid_joints(y, min_valid=-1e6):
    def and_all(x):
        if x.all():
            return 1
        return 0

    return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))


def mean_distance_error(y_true, y_pred):
    """Compute the mean distance error on predicted samples, considering
    only the valid joints from y_true.

    # Arguments
        y_true: [num_samples, nb_joints, dim]
        y_pred: [num_samples, nb_joints, dim]

    # Return
        The mean absolute error on valid joints.
    """

    assert y_true.shape == y_pred.shape
    num_samples = len(y_true)

    dist = np.zeros(y_true.shape[0:2])
    valid = np.zeros(y_true.shape[0:2])

    for i in range(num_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=1)

    match = dist * valid
    # print ('Maximum valid distance: {}'.format(match.max()))
    # print ('Average valid distance: {}'.format(match.mean()))

    return match.sum() / valid.sum()


def pckh(y_true, y_pred, valid, head_size, refp=0.5):
    """Compute the PCKh measure (using refp of the head size) on predicted
    samples.

    # Arguments
        y_true: [num_samples, nb_joints, 2]
        y_pred: [num_samples, nb_joints, 2]
        valid: [num_samples, nb_joints]
        head_size: [num_samples, 1]

    # Return
        The PCKh score.
    """

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(head_size)
    num_samples = len(y_true)

    # Ignore the joints 6 and 7 (pelvis and thorax respectively), according
    # to the file 'annolist2matrix.m'
    used_joints = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
    y_true = y_true[:, used_joints, :]
    y_pred = y_pred[:, used_joints, :]
    valid = valid[:, used_joints]
    head_size = np.tile(head_size, (1, len(used_joints)))

    dist = np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=-1))
    match = (dist / head_size <= refp) * valid

    return match.sum() / valid.sum()


def pckh_per_joint(y_true, y_pred, valid, head_size, joint_names,
        refp=0.5, verbose=1):
    """Compute the PCKh measure (using refp of the head size) on predicted
    samples per joint and output the results.

    # Arguments
        y_true: [num_samples, nb_joints, 2]
        y_pred: [num_samples, nb_joints, 2]
        vlaid: [num_samples, nb_joints]]
        head_size: [num_samples, 1]
        joint_names: Readable list of joint names.
    """

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(head_size)

    num_samples, num_joints = y_true.shape[:2]
    assert num_joints == len(joint_names), 'Invalid pose layout / joint_names'

    dist = np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=-1))

    for j in range(num_joints):
        jname = joint_names[j]
        space = 7*' '
        ss = len(space) - len(jname)
        if verbose:
            printc(jname + space[0:ss] + '| ', HEADER)
    if verbose:
        print ('')

    match = (dist / head_size <= refp) * valid
    for j in range(num_joints):
        pck = match[:, j].sum() / np.clip(valid[:, j].sum(), 1., None)
        if verbose:
            printc(' %.2f | ' % (100 * pck), OKBLUE)
    if verbose:
        print ('')

