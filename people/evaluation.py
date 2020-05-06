import numpy as np

from .metrics import mean_distance_error
from .metrics import pckh
from .metrics import pckh_per_joint
from .clustering import mean_confident_pose
from .utils import *


def average_world_poses(poses, meta, camera_indexes, conf=None):
    """meta composed of `a, s, e, c, f`"""

    if conf is not None:
        conf = np.expand_dims(conf, axis=-1).copy()
        # conf /= conf.max()
        # conf = np.power(conf, 2)

    idxhash = {}
    for i, m in enumerate(meta):
        if m[3] not in camera_indexes:
            continue

        """hash composed of `a, s, e, f`."""
        key = '%2d%2d%2d' % tuple(m[:3]) + '%2d' % m[4]
        if key not in idxhash:
            idxhash[key] = []
        idxhash[key].append(i)

    for key, idxs in idxhash.items():
        if conf is not None:
            p = 0
            cnt = 0
            for i in idxs:
                p += conf[i] * poses[i]
                cnt += conf[i]
            p /= cnt
        else:
            p = np.mean(poses[idxs], axis=0)
        for i in idxs:
            poses[i] = p

    return poses


def eval_human36m_mm_error(p_pred, z_pred, pose_w, afmat, scam, action,
        resol_z=2000.,
        map_to_pa17j=None,
        logdir=None,
        verbose=1):

    from .datasets.human36m import ACTION_LABELS
    from .datasets.human36m import ZBOUND
    from .datasets.human36m import MAX_Z

    assert len(p_pred) == len(z_pred) and \
            p_pred.ndim in [3, 4] and \
            z_pred.ndim in [1, 2] and \
            p_pred.ndim - z_pred.ndim == 2, (
                    'Invalid shape for p_pred ' + str(p_pred.shape)
                    + ' and/or for z_pred ' + str(z_pred.shape)
                    )

    afmat = afmat.copy()
    if p_pred.ndim == 4:
        multiperson = True
        num_samples, num_preds, num_joints, dimp1 = p_pred.shape
        p_pred = np.reshape(p_pred, (num_samples * num_preds, num_joints, dimp1))
        z_pred = np.reshape(z_pred, (num_samples * num_preds))
        afmat = np.expand_dims(afmat, axis=1)
        afmat = np.tile(afmat, (1, num_preds, 1, 1))
        afmat = np.reshape(afmat, (num_samples * num_preds, 3, 3))
    else:
        multiperson = False

    if verbose:
        printc('Avg. mm. error:', WARNING)

    p_pred = p_pred.copy()
    y_true_w = pose_w.copy()
    y_true_w = y_true_w[:, h36m23j3d.map_to_pa17j, :]
    y_pred_w = np.zeros(y_true_w.shape)

    """Move the root joints from g.t. poses to the origin."""
    y_true_w -= y_true_w[:, 0:1, :]

    """Project normalized coordiates to the image plane."""
    p_pred[:, :, 0:2] = transform_pose_sequence(
            afmat, p_pred[:, :, 0:2], inverse=True)

    """Recover the absolute Z."""
    z_pred = z_pred * (ZBOUND[1] - ZBOUND[0]) + ZBOUND[0]
    p_pred[:, :, 2] = (resol_z * (p_pred[:, :, 2] - 0.5)) \
            + np.expand_dims(z_pred, axis=-1)

    if multiperson:
        p_pred = np.reshape(p_pred,
                (num_samples, num_preds, num_joints, dimp1))
        new_uvd = np.zeros((num_samples, num_joints, dimp1))
        for i in range(len(p_pred)):
            new_uvd[i] = mean_confident_pose(p_pred[i])
        p_pred = new_uvd

    if map_to_pa17j is not None:
        y_pred_uvd = p_pred[:, map_to_pa17j, :3]
    else:
        y_pred_uvd = p_pred[:, :, :3]

    """Do the inverse camera projection."""
    for j in range(len(y_pred_uvd)):
        cam = camera_deserialize(scam[j])
        y_pred_w[j, :, :] = cam.inverse_project(y_pred_uvd[j])


    """Move the root joint from predicted poses to the origin."""
    y_pred_w[:, :, :] -= y_pred_w[:, 0:1, :]

    err_w = mean_distance_error(y_true_w[:, 0:, :], y_pred_w[:, 0:, :])
    if verbose:
        printc(' %.1f' % err_w, WARNING)

    if verbose:
        printcn('')

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        y_true_act[i] = None

    act = lambda x: action[x, 0]
    for i in range(len(y_pred_w)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = y_pred_w[i:i+1]
            y_true_act[act(i)] = y_true_w[i:i+1]
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], y_pred_w[i:i+1]], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], y_true_w[i:i+1]], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        err = mean_distance_error(y_true_act[i][:,0:,:], y_pred_act[i][:,0:,:])
        printcn('%s: %.1f' % (ACTION_LABELS[i], err), OKBLUE)

    printcn('(H36M) Final averaged error with estimated Z (mm): %.3f' \
            % err_w, WARNING)

    return err_w


def eval_human36m_activities(p_true, p_pred, action):

    from .datasets.human36m import ACTION_LABELS
    from .datasets.human36m import ZBOUND

    y_true_w = p_true #- p_true[:, 0:1, :]
    y_pred_w = p_pred #- p_pred[:, 0:1, :]

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        y_true_act[i] = None

    act = lambda x: action[x, 0]
    for i in range(len(y_pred_w)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = y_pred_w[i:i+1]
            y_true_act[act(i)] = y_true_w[i:i+1]
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], y_pred_w[i:i+1]], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], y_true_w[i:i+1]], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        err = mean_distance_error(y_true_act[i][:,0:,:], y_pred_act[i][:,0:,:])
        printcn('%s: %.1f' % (ACTION_LABELS[i], err), OKBLUE)



def eval_human36m_mm_error_model(model, x, aref, pose_w, afmat, scam, action,
        rootz=None,
        batch_size=8,
        map_to_pa17j=None,
        eval_cam_pred=False,
        logdir=None,
        verbose=1):

    from .datasets.human36m import ACTION_LABELS
    from .datasets.human36m import ZBOUND
    from .datasets.human36m import BBOX_REF
    from .datasets.human36m import MAX_Z

    inputs = [x, aref]
    num_blocks = len(model.outputs) - 1 - int(eval_cam_pred)

    lower_err = np.inf
    lower_i = -1
    scores = []
    anchor_id = 0

    y_true_w = pose_w.copy()
    # y_true_w = y_true_w[:, h36m23j3d.map_to_pa17j, :]
    y_pred_w = np.zeros((num_blocks,) + y_true_w.shape)

    pred = model.predict(inputs, batch_size=batch_size, verbose=1)
    # z_pred = np.mean(ZBOUND) / np.clip(pred[0], 0.0001, None)
    z_pred = MAX_Z * pred[0]
    z_pred = np.clip(z_pred, ZBOUND[0], ZBOUND[1])
    del pred[0]

    abs_z_err = np.inf
    if rootz is not None:
        # rootz = np.mean(ZBOUND) / np.clip(rootz, 0.0001, None)
        rootz = MAX_Z * rootz
        y_pred_w_gtz = np.zeros((num_blocks,) + y_true_w.shape)
        print ('zrootz', rootz.min(), rootz.max(), rootz.mean())
        print ('z_pred', z_pred.min(), z_pred.max(), z_pred.mean())
        abs_z_err = np.abs(rootz - z_pred)
        print ('abs_z_err', abs_z_err.min(), abs_z_err.max(), abs_z_err.mean())
        abs_z_err = abs_z_err.mean()

    """Move the root joints from g.t. poses to the origin."""
    y_true_w -= y_true_w[:, 0:1, :]

    if verbose:
        printc('Avg. mm. error:', WARNING)

    for b in range(num_blocks):
        y_pred = pred[b][:, :, 0:3]

        """Project normalized coordiates to the image plane."""
        y_pred[:, :, 0:2] = transform_pose_sequence(
            afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        """Recover the absolute Z."""
        if rootz is not None:
            y_pred_gtz = y_pred.copy()
            y_pred_gtz[:, :, 2] = \
                    (BBOX_REF * (y_pred_gtz[:, :, 2] - 0.5)) + rootz

        y_pred[:, :, 2] = (BBOX_REF * (y_pred[:, :, 2] - 0.5)) + z_pred

        y_pred_uvd = y_pred[:, :, 0:3]
        if rootz is not None:
            y_pred_uvd_gtz = y_pred_gtz[:, :, 0:3]

        """Do the inverse camera projection."""
        for j in range(len(y_pred_uvd)):
            cam = camera_deserialize(scam[j])
            y_pred_w[b, j, :, :] = cam.inverse_project(y_pred_uvd[j])
            if rootz is not None:
                y_pred_w_gtz[b, j, :, :] = \
                        cam.inverse_project(y_pred_uvd_gtz[j])

        """Move the root joint from predicted poses to the origin."""
        y_pred_w[b, :, :, :] -= y_pred_w[b, :, 0:1, :]
        if rootz is not None:
            y_pred_w_gtz[b, :, :, :] -= y_pred_w_gtz[b, :, 0:1, :]

        err_w = mean_distance_error(y_true_w[:, 0:, :], y_pred_w[b, :, 0:, :])
        scores.append(err_w)
        if verbose:
            printc(' %.1f' % err_w, WARNING)

        if rootz is not None:
            err_w_gtz = mean_distance_error(y_true_w[:, 0:, :],
                    y_pred_w_gtz[b, :, 0:, :])
            if verbose:
                printc(' %.1f' % err_w_gtz, FAIL)

        """Keep the best prediction and its index."""
        if err_w < lower_err:
            lower_err = err_w
            lower_i = b

    if verbose:
        printcn('')

    if logdir is not None:
        np.save('%s/h36m_y_pred_w.npy' % logdir, y_pred_w)
        np.save('%s/h36m_y_true_w.npy' % logdir, y_true_w)

    """Select only the best prediction."""
    y_pred_w = y_pred_w[lower_i]
    if rootz is not None:
        y_pred_w_gtz = y_pred_w_gtz[lower_i]

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    if rootz is not None:
        y_pred_act_gtz = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        if rootz is not None:
            y_pred_act_gtz[i] = None
        y_true_act[i] = None

    act = lambda x: action[x, 0]
    for i in range(len(y_pred_w)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = y_pred_w[i:i+1]
            if rootz is not None:
                y_pred_act_gtz[act(i)] = y_pred_w_gtz[i:i+1]
            y_true_act[act(i)] = y_true_w[i:i+1]
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], y_pred_w[i:i+1]], axis=0)
            if rootz is not None:
                y_pred_act_gtz[act(i)] = np.concatenate(
                        [y_pred_act_gtz[act(i)], y_pred_w_gtz[i:i+1]], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], y_true_w[i:i+1]], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        err = mean_distance_error(y_true_act[i][:,0:,:], y_pred_act[i][:,0:,:])
        if rootz is not None:
            err_gtz = mean_distance_error(y_true_act[i][:,0:,:],
                    y_pred_act_gtz[i][:,0:,:])
        if rootz is not None:
            printcn('%s: %.1f (%.1f)' % (ACTION_LABELS[i], err, err_gtz),
                    OKBLUE)
        else:
            printcn('%s: %.1f' % (ACTION_LABELS[i], err), OKBLUE)

    printcn('(H36M) Final averaged error with estimated Z (mm): %.3f' \
            % lower_err, WARNING)
    printcn('(H36M) Final absolute Z error (mm): {}'.format(abs_z_err),
            OKGREEN)

    return scores


def eval_mpii_pckh(model, fval, arefval, pval, afmat_val, headsize_val,
        batch_size=8,
        refp=0.5,
        map_to_mpii=None,
        verbose=1):
    """Evaluate MPII score using pckh.

    # Arguments
        model: Keras model
        fval: [num_samples, num_anchors_size, num_joints, dim+1]
        arefval: [num_samples, num_anchors, 4]
        pval: [num_samples, num_anchors, num_joints, dim+1]
        afmat_val: [num_samples, 3, 3]
        headsize_val: [num_samples, 1]
    """

    num_blocks = len(model.outputs) - 1
    inputs = [fval, arefval]
    scores = []

    pred = model.predict(inputs, batch_size=batch_size, verbose=1)
    del pred[0]

    A = afmat_val[:]
    if map_to_mpii is not None:
        y_true = pval[:, map_to_mpii]
    else:
        y_true = pval

    y_true = transform_pose_sequence(A.copy(), y_true[..., :2], inverse=True)
    valid = (y_true[..., -1] > 0.5).astype(float)

    if verbose:
        printc('PCKh on validation:', WARNING)

    for b in range(num_blocks):

        if map_to_mpii is not None:
            y_pred = pred[b][:, map_to_mpii, :2]
        else:
            y_pred = pred[b][:, :, :2]

        y_pred = transform_pose_sequence(A.copy(), y_pred, inverse=True)
        s = pckh(y_true, y_pred, valid, headsize_val, refp=refp)
        if verbose:
            printc(' %.1f' % (100*s), WARNING)
        scores.append(s)

        if b == num_blocks-1:
            if verbose:
                printcn('')
            pckh_per_joint(y_true, y_pred, valid, headsize_val,
                    mpii16j.joint_names, verbose=verbose)

    return scores


def eval_mpii3dhp_mm_error(model, x, aref, pose_w, afmat, sub,
        rootz=None,
        resol_z=2000.,
        batch_size=8,
        map_to_pa17j=None,
        map_hflip=None,
        logdir=None,
        verbose=1):

    from .datasets.mpii3dhp import ACTION_LABELS
    from .datasets.mpii3dhp import ZBOUND
    from .datasets.mpii3dhp import inverse_projection_te
    from .datasets.mpii3dhp import MAX_Z

    inputs = [x, aref]
    num_blocks = len(model.outputs) - 1

    lower_err = np.inf
    lower_i = -1
    scores = []

    y_true_w = pose_w.copy()
    y_pred_w = np.zeros((num_blocks,) + y_true_w.shape)

    pred = model.predict(inputs, batch_size=batch_size, verbose=1)
    z_pred = MAX_Z * pred[0]
    del pred[0]

    if rootz is not None:
        z_pred = rootz

    """Move the root joints from g.t. poses to the origin."""
    y_true_w_ref = y_true_w - y_true_w[:,0:1,:]

    if verbose:
        printc('Avg. mm. error:', WARNING)

    for b in range(num_blocks):
        if map_to_pa17j is not None:
            y_pred = pred[b][:, map_to_pa17j, 0:3]
        else:
            y_pred = pred[b][:, :, 0:3]

        if map_hflip is not None:
            y_pred = y_pred[:, map_hflip, :3]

        """Project normalized coordiates to the image plane."""
        y_pred[:, :, 0:2] = transform_pose_sequence(
            afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        """Recover the absolute Z."""
        y_pred[:, :, 2] = (resol_z * (y_pred[:, :, 2] - 0.5)) + z_pred

        """Recover pose uvd."""
        y_pred_uvd = y_pred[:, :, 0:3]

        for k in range(len(y_pred_uvd)):
            y_pred_w[b, k] = inverse_projection_te(y_pred_uvd[k], sub[k])

        """Move the root joint from predicted poses to the origin."""
        y_pred_w_ref = y_pred_w[b, :, :, :] - y_pred_w[b, :, 0:1, :]

        err_w = mean_distance_error(y_true_w_ref, y_pred_w_ref)
        scores.append(err_w)
        if verbose:
            printc(' %.1f' % err_w, WARNING)

        """Keep the best prediction and its index."""
        if err_w < lower_err:
            lower_err = err_w
            lower_i = b

    if verbose:
        printcn('')

    if logdir is not None:
        np.save('%s/mpiinf3dhp_y_pred_w.npy' % logdir, y_pred_w)
        np.save('%s/mpiinf3dhp_y_true_w.npy' % logdir, y_true_w)

    printcn('(MPI-INF-3DHP) Final averaged error (mm): %.3f' \
            % lower_err, WARNING)

    # return scores, y_true_w, y_pred_w
    return scores


def eval_mpii3dhp_mm_pck3d(p_true, p_pred, action):

    from people.datasets.mpii3dhp import ACTION_LABELS

    dist = np.sqrt(np.sum(np.square(p_true - p_pred), axis=-1))
    err = np.mean(dist)
    print ('Mean error', err)
    print (dist.shape)
    pck = np.sum(dist < 150) / np.sum(dist > 0)
    print ('PCK', pck)

    # from sklearn import metrics

    auc = 0
    x = []
    y = []
    for t in range(5,145,5):
        pck = np.sum(dist < t) / np.sum(dist > 0)
        print (pck)
        x.append(t)
        y.append(pck)

    x = np.array(x)
    y = np.array(y)
    auc = np.sum(x*y) / np.sum(x)

    # auc = metrics.auc(x, y)

    print ('AUC', auc)

    y_pred_w = p_pred
    y_true_w = p_true

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        y_true_act[i] = None

    act = lambda x: action[x, 0]
    for i in range(len(y_pred_w)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = y_pred_w[i:i+1]
            y_true_act[act(i)] = y_true_w[i:i+1]
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], y_pred_w[i:i+1]], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], y_true_w[i:i+1]], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        dist = np.sqrt(np.sum(np.square(y_true_act[i] - y_pred_act[i]), axis=-1))
        err = np.mean(dist)
        printcn(OKBLUE, '%s: %.1f' % (ACTION_LABELS[i], err))

        pck = np.sum(dist < 150) / np.sum(dist > 0)
        print ('PCK', pck)


