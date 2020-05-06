import os

import numpy as np
import scipy.io as sio
import h5py
import json
from PIL import Image

from .generic import GenericDataset
from .generic import project_gt_poses_to_anchors
from .generic import compute_window_reference
from ..utils import *


# Define focal length for testing cameras
focal_len_te = {1: 7.32506}
focal_len_te[4] = focal_len_te[3] = focal_len_te[2] = focal_len_te[1]
focal_len_te[6] = focal_len_te[5] = 8.770747185

# Define center offset in mm for testing cameras
offsetx_te = {1: -0.0322884}
offsetx_te[4] = offsetx_te[3] = offsetx_te[2] = offsetx_te[1]
offsetx_te[6] = offsetx_te[5] = -0.104908645

offsety_te = {1: 0.0929296}
offsety_te[4] = offsety_te[3] = offsety_te[2] = offsety_te[1]
offsety_te[6] = offsety_te[5] = 0.104899704

# Define image size (width, height) in px for testing cameras
imgsize_te = {1: (2048, 2048)}
imgsize_te[4] = imgsize_te[3] = imgsize_te[2] = imgsize_te[1]
imgsize_te[6] = imgsize_te[5] = (1920, 1080)

# This will be filled with K coef. for testing samples estimated from data
ksx_te = {}
ksy_te = {}

mpii3dhp_joint_names = [
        'spine3',            # 0
        'spine4',            # 1
        'spine2',            # 2
        'spine',             # 3
        'pelvis',            # 4
        'neck',              # 5
        'head',              # 6
        'head_top',          # 7
        'left_clavicle',     # 8
        'left_shoulder',     # 9
        'left_elbow',        # 10
        'left_wrist',        # 11
        'left_hand',         # 12
        'right_clavicle',    # 13
        'right_shoulder',    # 14
        'right_elbow',       # 15
        'right_wrist',       # 16
        'right_hand',        # 17
        'left_hip',          # 18
        'left_knee',         # 19
        'left_ankle',        # 20
        'left_foot',         # 21
        'left_toe',          # 22
        'right_hip',         # 23
        'right_knee',        # 24
        'right_ankle',       # 25
        'right_foot',        # 26
        'right_toe'          # 27
        ]

# [-442.59605121 7399.46575735]
ZBOUND = np.array([0, 7399.46575735])
MAX_Z = 8000
BBOX_REF = 2000
ACTION_LABELS = ['Walking/Standing', 'Exercise', 'Sitting(1)', 'Crouch/Reach',
        'On the Floor', 'Sports', 'Sitting(2)', 'Miscellaneous']


def str2float(s):
    for i in range(len(s)):
        s[i] = float(s[i])

    return s


def read_camera_calibration(fname):
    calib = {}
    try:
        fid = open(fname, 'r')
        header = fid.readline()

        while True:
            line = fid.readline()
            if line == '':
                break

            name = int(line[14:-1])
            calib[name] = {}

            line = fid.readline()[14:-1].split(' ')
            calib[name]['sensor'] = np.array(str2float(line)).tolist()

            line = fid.readline()[14:-1].split(' ')
            calib[name]['size'] = np.array(str2float(line)).tolist()

            animated = fid.readline()

            line = fid.readline()[14:-2].split(' ')
            calib[name]['intrinsic'] = np.reshape(
                    str2float(line), (4, 4)).tolist()

            line = fid.readline()[14:-2].split(' ')
            calib[name]['extrinsic'] = np.reshape(
                    str2float(line), (4, 4)).tolist()

            radial = fid.readline()
    except Exception as e:
        warning('Error while reading file {}\n >> {}'.format(fname, e))

    return calib


def convert_matfiles(dataset_path,
        subjects=range(1, 8+1),
        seqs=range(1, 2+1),
        cameras=range(9),
        out_filename='annot3d.json'):

    annot = {}

    for sub in subjects:
        for seq in seqs:
            print (sub, seq)

            annfile = 'S{}/Seq{}/annot.mat'.format(sub, seq)
            mat = sio.loadmat(dataset_path + '/' + annfile)
            annot2 = mat['annot2']
            annot3 = mat['annot3']

            calibfile = 'S{}/Seq{}/camera.calibration'.format(sub, seq)
            calib = read_camera_calibration(dataset_path + '/' + calibfile)

            for cam in cameras: # Load the selected cameras
                ann2 = np.reshape(annot2[cam][0], (-1, 28, 2))
                ann3 = np.reshape(annot3[cam][0], (-1, 28, 3))
                # Select the 17 joints compatible with Human3.6M PA17j layout
                # ann2 = ann2[:, pa17j3d.map_from_mpii3dhp, :].tolist()
                # ann3 = ann3[:, pa17j3d.map_from_mpii3dhp, :].tolist()
                ann2 = ann2.tolist()
                ann3 = ann3.tolist()

                key = '%d:%d:%d' % (sub, seq, cam)
                print ('  >> ', key)
                annot[key] = {}
                annot[key]['calib'] = calib[cam]
                annot[key]['annot2'] = ann2
                annot[key]['annot3'] = ann3

    with open('{}/{}'.format(dataset_path, out_filename), 'w') as fp:
            json.dump(annot, fp)


def convert_matfiles_test(dataset_path,
        subjects=range(1, 6+1),
        out_filename='annot3d_test.json'):

    annot = {}

    for sub in subjects:

        annfile = 'mpi_inf_3dhp_test_set/TS{}/annot_data.mat'.format(sub)
        with h5py.File(dataset_path + '/' + annfile, 'r') as f:
            key = '%d' % sub
            annot[key] = {}
            ann2 = np.squeeze(f['annot2'])[:, pa17j3d.map_from_mpii3dhp_te, :]
            ann3 = np.squeeze(f['annot3'])[:, pa17j3d.map_from_mpii3dhp_te, :]
            annot[key]['annot2'] = ann2.tolist()
            annot[key]['annot3'] = ann3.tolist()
            annot[key]['valid_frame'] = np.squeeze(f['valid_frame']).tolist()
            annot[key]['activity'] = \
                    np.squeeze(f['activity_annotation']).tolist()

    with open('{}/{}'.format(dataset_path, out_filename), 'w') as fp:
            json.dump(annot, fp)


def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def estimate_k(p2d, p3d, ts):
    x2 = p2d[:, 0]
    y2 = p2d[:, 1]
    x3 = p3d[:, 0]
    y3 = p3d[:, 1]
    z3 = p3d[:, 2]
    kx = (x2 - imgsize_te[ts][0]/2) / (focal_len_te[ts]*x3/z3 + offsetx_te[ts])
    ky = (y2 - imgsize_te[ts][1]/2) / (focal_len_te[ts]*y3/z3 + offsety_te[ts])

    kx = np.mean(reject_outliers(kx))
    ky = np.mean(reject_outliers(ky))

    return kx, ky


def inverse_projection_te(pred_uvd, ts):
    assert pred_uvd.ndim == 2 and pred_uvd.shape[-1] == 3, ('Invalid pred_uvd,'
            ' expecting a tensor with shape (num_points, 3),'
            ' but got {}'.format(pred_uvd.shape))
    assert ts in [1, 2, 3, 4, 5, 6], 'Invalid `test subject` index.'
    ts = int(ts)

    f = focal_len_te[ts]
    ox = offsetx_te[ts]
    oy = offsety_te[ts]
    w, h = imgsize_te[ts]
    kx = ksx_te[ts]
    ky = ksy_te[ts]

    pw = pred_uvd.copy()
    pw[:, 0] = (pw[:, 2] / f) * ((pw[:, 0] - w / 2) / kx - ox)
    pw[:, 1] = (pw[:, 2] / f) * ((pw[:, 1] - h / 2) / ky - oy)

    return pw


def load_mpiinf3d_json_annotations(filename, mode, verbose=1):

    if verbose:
        printcn(OKBLUE,
                'MPI-INF-3DHP: Loading annotation file {}'.format( filename))

    sequence = []
    with open(filename, 'r') as fp:
        data = json.load(fp)

    if mode == TEST_MODE:
        for sub in range(1, 6+1):
            frame_seq = data['%d' % sub]
            valid = np.array(frame_seq['valid_frame'])
            idx_valid = np.where(valid == 1)[0]
            annot2 = np.array(frame_seq['annot2'])[idx_valid]
            annot3 = np.array(frame_seq['annot3'])[idx_valid]
            activity = np.array(frame_seq['activity'])[idx_valid]
            sequence.append((annot2, annot3, activity, sub, idx_valid))

            kxs = []
            kys = []
            for p2d, p3d in zip(annot2, annot3):
                kx, ky = estimate_k(p2d, p3d, sub)
                kxs.append(kx)
                kys.append(ky)

            kxs = reject_outliers(np.array(kxs))
            kys = reject_outliers(np.array(kys))

            ksx_te[sub] = np.mean(kxs)
            ksy_te[sub] = np.mean(kys)
            if verbose:
                printnl('S{}, K(x): {} avg, {} std'.format(
                    sub, ksx_te[sub], np.std(kxs)))
                printnl('S{}, K(y): {} avg, {} std'.format(
                    sub, ksy_te[sub], np.std(kys)))

    elif mode == TRAIN_MODE:
        for sub in [1, 2, 3, 4, 5, 6, 7, 8]:
            for seq in [1, 2]:
                for cam in [0, 1, 2, 4, 5, 6, 7, 8]:
                    key = '%d:%d:%d' % (sub, seq, cam)
                    if key not in data.keys():
                        continue # Ignore if key not present

                    frame_seq = data[key]

                    calib = frame_seq['calib']
                    annot2 = np.array(frame_seq['annot2'])[:-2]
                    annot3 = np.array(frame_seq['annot3'])[:-2]
                    sequence.append((annot2, annot3, calib, key))

                    del data[key] # Free memory as soon as possible

    return sequence


class MpiInf3D(GenericDataset):
    """Implementation of the MPI-INF-3D Human Pose dataset for 3D human pose
    estimation.
    """

    def __init__(self, dataset_path, dataconf,
            poselayout=pa17j3d,
            remove_outer_joints=True,
            preprocess_mode='tf',
            recompute_zbound=False):

        GenericDataset.__init__(self, dataset_path, dataconf, poselayout,
                remove_outer_joints=remove_outer_joints,
                preprocess_mode=preprocess_mode)

        self.load_annotations(
                os.path.join(dataset_path, 'annot3d.json'),
                os.path.join(dataset_path, 'annot3d_test.json'),
                recompute_zbound)

    def load_annotations(self, annot_filename_tr, annot_filename_te,
            recompute_zbound):

        try:
            seq_te = load_mpiinf3d_json_annotations(annot_filename_te,
                    TEST_MODE)
            print ('length of seq_te: ', len(seq_te))


            frames_idx_te = serialize_index_sequences(seq_te)

            seq_tr = load_mpiinf3d_json_annotations(annot_filename_tr,
                    TRAIN_MODE)
            print ('length of seq_tr: ', len(seq_tr))

            frames_idx_tr = serialize_index_sequences(seq_tr)

            self.sequences = [seq_te, seq_tr, []]
            self.frame_idx = [frames_idx_te, frames_idx_tr, []]

            if recompute_zbound:
                zarray = np.zeros((len(self.frame_idx[TRAIN_MODE]),))
                idx = 0
                warning('Recomputing Z-boundary for MPI-INF-3DHP!')
                for seq in self.sequences[TRAIN_MODE]:
                    roots = seq[1][:, 4, 2]
                    zarray[idx:idx+len(roots)] = roots
                    idx += len(roots)
                zarray[zarray < -1e6] = np.nan
                avg = np.nanmean(zarray)
                zmax = np.nanmax(zarray)
                zmin = np.nanmin(zarray)
                warning('avg {}, max {}, min {}'.format(avg, zmax, zmin))
                margin = 1.05 * max(avg - zmin, zmax - avg)
                self.zbound = np.array([avg - margin, avg + margin])
                warning('zbound: {}'.format(self.zbound))
            else:
                self.zbound = ZBOUND

        except:
            warning('Error loading MPI-INF-3DHP dataset!')
            raise

    def get_data(self, key, mode):
        pl = self.poselayout # alias for poselayout
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
            img_div = 2
        else:
            dconf = self.dataconf.get_fixed_config()
            img_div = 1

        seq_idx, frame_idx = self.frame_idx[mode][key]
        frame_seq = self.sequences[mode][seq_idx]

        p2d = frame_seq[0][frame_idx]
        p3d = frame_seq[1][frame_idx]
        pose_uvd = np.concatenate([p2d[:, 0:2] / img_div, p3d[:, 2:3]], axis=-1)

        if mode == TRAIN_MODE:
            activity = 0
            calib = frame_seq[2]
            sub, seq, cam = [int(s) for s in frame_seq[3].split(':')]
            fidx = frame_idx + 1
            root_joint = pose_uvd[4:5, :]
            image = 'images/S%d/Seq%d/img_%d_%06d.jpg' % (sub, seq, cam, fidx)

        else:
            activity = frame_seq[2][frame_idx]
            sub = frame_seq[3]
            fidx = frame_seq[4][frame_idx] + 1
            root_joint = pose_uvd[0:1, :]
            image = 'mpi_inf_3dhp_test_set/TS%d/imageSequence/img_%06d.jpg' \
                            % (sub, fidx)

        output['sub'] = sub
        output['action'] = activity - 1

        """Load image frame."""
        imgt = T(Image.open(os.path.join(self.dataset_path, image)))
        imgsize = imgt.size

        """We approximate the focal length for all cameras, since it is
        used just to define a rough cropping size.
        """
        objpos, winsize, zrange = get_crop_params(root_joint[0], imgt.size,
                scale=dconf['scale'])

        yshift = max(self.dataconf.crop_resolution) // 11
        objpos += dconf['scale'] \
                * np.array([dconf['transx'], yshift + dconf['transy']])

        imgt.rotate_crop(dconf['angle'], objpos, winsize)
        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.resize(self.dataconf.crop_resolution)
        imgt.normalize_affinemap()
        frame = normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower'], mode=self.preprocess_mode)

        if dconf['geoocclusion'] is not None:
            geo = dconf['geoocclusion']
            frame[geo[0]:geo[2], geo[1]:geo[3], :] = 0.

        output['frame'] = frame

        """Project pose to cropped region."""
        tpose = np.empty(pose_uvd.shape)
        tpose[:, 0:2] = transform_2d_points(imgt.afmat, pose_uvd[:, 0:2],
                transpose=True)
        tpose[:, 2] = (pose_uvd[:, 2] - zrange[0]) / (zrange[1] - zrange[0])
        vis = get_visible_joints(tpose[:, 0:2])

        """Handle the pose/skeleton layout."""
        if (pl == mpiinf3d28j3d) or (mode == TEST_MODE):
            p = tpose
            c = np.expand_dims(vis, axis=-1)

        elif pl == pa17j3d:
            p = tpose[pl.map_from_mpii3dhp, :]
            c = np.expand_dims(vis[pl.map_from_mpii3dhp], axis=-1)

        else:
            p = np.nan * np.ones((pl.num_joints, pl.dim))
            c = np.nan * np.ones((pl.num_joints, 1))
            p[pl.map_to_mpii3dhp, 0:pl.dim] = tpose[:, 0:pl.dim]
            c[pl.map_to_mpii3dhp, 0] = vis

        p = np.concatenate((p, c), axis=-1)
        if imgt.hflip:
            if mode == TRAIN_MODE:
                p = p[pl.map_hflip, :]
            else:
                p = p[pa17j3d.map_hflip, :]

        if self.remove_outer_joints:
            p[p < 0] = -1
            p[p > 1] = -1

        """Set NaN values as an invalid value (-1e9)."""
        p[np.isnan(p)] = -1e9

        """Save afmat and compute the `aref` field."""
        afmat = imgt.afmat.copy()
        output['afmat'] = afmat
        output['aref'] = compute_window_reference(afmat, imgsize)

        """Normalize the absolute Z for root joint, and keep its value only if
        it is visible in each anchor.
        """
        # rootz = np.mean(self.zbound) \
                # / np.clip(root_joint[0, 2], self.zbound[0], self.zbound[1])
        rootz = root_joint[0, 2] / MAX_Z
        output['rootz'] = rootz

        if mode == TRAIN_MODE:
            output['pose_w'] = p3d
            output['pose'] = p
        else:
            output['pose_w_te'] = p3d
            output['pose_te'] = p

        return output

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.dataconf.input_shape
        if dictkey == 'pose':
            return (self.poselayout.num_joints, self.poselayout.dim + 1)
        if dictkey == 'pose_te':
            return (pa17j3d.num_joints, pa17j3d.dim+1)
        if dictkey == 'pose_w':
            return (mpiinf3d28j3d.num_joints, mpiinf3d28j3d.dim)
        if dictkey == 'pose_w_te':
            return (pa17j3d.num_joints, pa17j3d.dim)
        if dictkey == 'rootz':
            return (1, )
        if dictkey == 'aref':
            return (4, )
        if dictkey == 'action':
            return (1,)
        if dictkey == 'sub':
            return (1,)
        if dictkey == 'camera':
            return (21,)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey `{}` on get_shape!'.format(dictkey))

    def get_length(self, mode):
        return len(self.frame_idx[mode])


def get_crop_params(rootjoint, imgsize, scale):

    objpos = np.array([rootjoint[0], rootjoint[1] + scale*1])
    d = rootjoint[2]
    winsize = 1700*scale*max(imgsize[0], imgsize[1])/d
    zrange = np.array([d - scale*1000., d + scale*1000.])

    return objpos, (winsize, winsize), zrange


def serialize_index_sequences(frame_seq):
    frames_idx = []
    for s in range(len(frame_seq)):
        for f in range(len(frame_seq[s][0])):
            frames_idx.append((s, f))

    print ('Total of frames', len(frames_idx))

    return frames_idx

def get_crop_params_old(rootjoint, imgsize, scale):
    print ('DAMNED mpiinf3d scale', scale)

    objpos = np.array([rootjoint[0], rootjoint[1] + scale*1])
    d = rootjoint[2]
    winsize = 1800 * scale * max(imgsize[0], imgsize[1]) / d
    zrange = np.array([d - scale*1000., d + scale*1000.])

    return objpos, (winsize, winsize), zrange
