import os

import json
import numpy as np
import scipy.io as sio
from PIL import Image

from .generic import GenericDataset
from .generic import project_gt_poses_to_anchors
from .generic import compute_anchors_reference
from .generic import compute_window_reference
from ..utils import *

ACTION_LABELS = None
ZBOUND = np.array([2378.56192888, 7916.5468051])
MAX_Z = 8000
BBOX_REF = 2000


def load_h36m_mat_annotation(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Respect the order of TEST (0), TRAIN (1), and VALID (2)
    sequences = [mat['sequences_te'], mat['sequences_tr'], mat['sequences_val']]
    action_labels = mat['action_labels']
    joint_labels = mat['joint_labels']

    return sequences, action_labels, joint_labels


def load_h36m_mat_calib_test(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    sequences = mat['sequences_te']
    action_labels = mat['action_labels']
    joint_labels = mat['joint_labels']

    return sequences, action_labels, joint_labels


def serialize_index_sequences(seq):
    frames_idx = []
    for s in range(len(seq)):
        for f in range(len(seq[s].frames)):
            frames_idx.append((s, f))

    return frames_idx


def parse_json_bbox(fname):
    with open(fname, 'r') as fid:
        data = json.load(fid)
    bbox_pred = np.zeros((len(data), 4))
    for i in range(len(data)):
        p = data['%d' % i]
        if len(p) == 0:
            p = [[200, 200, 800, 800]]

        obj, win = bbox_to_objposwin(p[0], square=True)
        bbox_pred[i] = objposwin_to_bbox(obj, 1.25*win)

    return bbox_pred



class Human36M(GenericDataset):
    """Implementation of the Human3.6M dataset for 3D pose estimation, for
    training and validation splits.
    """

    def __init__(self, dataset_path, dataconf,
            bbox_crop_mode='3d',
            remove_outer_joints=True,
            preprocess_mode='tf',
            recompute_zbound=False,
            zbound=None,
            bbox_file_train=None,
            bbox_file_val=None,
            pose_pred_train=None,
            pose_pred_val=None):
        """Instanciates the class Human36M.

        How the bounding box is cropped:
            First, it checks for pose predictions (given by
            pose_pred_train/val). If available, it is used to crop the 3D
            bounding box. If not, it checks for the files bbox_file_train/val.
            If the file was given, it is used on validation. On training, the
            given bboxes are used with probability `bbox_ratio_train`, and
            ground truth 3D bounding boxes are used otherwise. If bbox_files
            are not given, only ground truth 3D bboxes are used.

        # Arguments

            dataset_path: string. Path to the Human3.6M dataset.
            dataconf: object DataConfig.
            bbox_crop_mode: string '2d', '3d' or None. Define the mode to crop
                each frame, considering the 2d points in the image plane, the
                3d (u-v plus depth) for a 2^3 meters cube, or None for full
                frame.
            remove_outer_joints: boolean. Remove outer body joints (from the
            bounding box) or not.
            preprocess_mode: string.
            recompute_zbound: boolean. If True, recompute the bounding limits
                for absolute z.
            zbound: list of floats. Minimum and maximim values for absolute z.
            bbox_file_train and bbox_file_val: strings. Path to train/val json
                files containing 2d bounding boxes. It is replaced by the
                prediction pose if that is given.
            pose_pred_train and pose_pred_val: string or None. Path to a numpy
                file with previous pose predictions. These predictions, if
                geven, will be used to crop the bounding boxes. When not given,
                bounding boxes are cropped based on the ground truth poses.
                Predictions should be in the UVD format.
        """
        GenericDataset.__init__(self,
                dataset_path,
                dataconf,
                poselayout=pa17j3d,
                remove_outer_joints=remove_outer_joints,
                preprocess_mode=preprocess_mode)

        self.bbox_crop_mode = bbox_crop_mode
        self.bbox_pred = 3*[None]
        self.pose_pred = 3*[None]


        if bbox_file_train is not None:
            self.bbox_pred[TRAIN_MODE] = parse_json_bbox(bbox_file_train)

        if bbox_file_val is not None:
            self.bbox_pred[VALID_MODE] = parse_json_bbox(bbox_file_val)

        if pose_pred_train is not None:
            self.pose_pred[TRAIN_MODE] = np.load(pose_pred_train)

        if pose_pred_val is not None:
            self.pose_pred[VALID_MODE] = np.load(pose_pred_val)

        self._load_annotations(os.path.join(dataset_path, 'annotations.mat'),
                recompute_zbound, zbound=zbound)


    def _load_annotations(self, filename, recompute_zbound, zbound=None):
        try:
            self.sequences, self.action_labels, self.joint_labels = \
                    load_h36m_mat_annotation(filename)
            self.frame_idx = [serialize_index_sequences(self.sequences[0]),
                    serialize_index_sequences(self.sequences[1]),
                    serialize_index_sequences(self.sequences[2])]

            global ACTION_LABELS
            ACTION_LABELS = self.action_labels

            if recompute_zbound:
                zarray = np.zeros((len(self.frame_idx[TRAIN_MODE]),))
                idx = 0
                warning('Recomputing Z-boundary for Human3.6M!')
                for seq in self.sequences[TRAIN_MODE]:
                    cpar = seq.camera_parameters
                    cam = Camera(cpar.R, cpar.T, cpar.f, cpar.c, cpar.p, cpar.k)
                    pw = self.load_sequence_pose_annot(seq.frames)
                    roots = cam.project(pw[:, 0, :])
                    zarray[idx:idx+len(roots)] = roots[:, 2] / np.mean(cam.f)
                    idx += len(roots)
                zarray[zarray < -1e6] = np.nan
                avg = np.nanmean(zarray)
                zmax = np.nanmax(zarray)
                zmin = np.nanmin(zarray)
                warning('avg {}, max {}, min {}'.format(avg, zmax, zmin))
                margin = 1.05 * max(avg - zmin, zmax - avg)
                self.zbound = np.array([avg - margin, avg + margin])
                warning('zbound: {}'.format(self.zbound))

            elif zbound is not None:
                self.zbound = zbound

            else:
                self.zbound = ZBOUND

        except:
            warning('Error loading Human3.6M dataset!')
            raise

    def get_meta(self, key, mode):
        seq_idx, frame_idx = self.frame_idx[mode][key]
        seq = self.sequences[mode][seq_idx]
        a = int(seq.name[1:3])
        s = int(seq.name[5:7])
        e = int(seq.name[9:11])
        c = int(seq.name[13:15])
        f = seq.frames[frame_idx].f

        return (a, s, e, c, f)


    def get_data(self, key, mode, frame_list=None):
        pl = self.poselayout # alias for poselayout
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
            random_clip = True
        else:
            dconf = self.dataconf.get_fixed_config()
            random_clip = False

        seq_idx, frame_idx = self.frame_idx[mode][key]
        seq = self.sequences[mode][seq_idx]
        objframe = seq.frames[frame_idx]

        """Build a Camera object"""
        cpar = seq.camera_parameters
        cam = Camera(cpar.R, cpar.T, cpar.f, cpar.c, cpar.p, cpar.k)

        """Load and project pose into the camera coordinates."""
        pose_w = objframe.pose3d.T[h36m23j3d.map_from_h36m, 0:h36m23j3d.dim]
        pose_w = pose_w[h36m23j3d.map_to_pa17j]

        taux = T(None, img_size=(1, 1))
        taux.rotate_center(dconf['angle'])
        if dconf['hflip'] == 1:
            taux.horizontal_flip()

        pose_uvd = cam.project(pose_w, project_from_world=True)

        imgsize = (objframe.w, objframe.h)

        """Compute bounding box."""
        bbox_pred = pose_pred = None
        if self.bbox_pred[mode] is not None:
            bbox_pred = self.bbox_pred[mode][key]
        if self.pose_pred[mode] is not None:
            pose_pred = self.pose_pred[mode][key]

        objpos, winsize, zrange = auto_bbox_cropping(
                gt_pose_uvd=pose_uvd,
                focal=cam.f,
                box_size_mm=BBOX_REF,
                dconf=dconf,
                imgsize=imgsize,
                bbox_crop_mode=self.bbox_crop_mode,
                bbox_pred=bbox_pred,
                pose_pred=pose_pred,
                mode=mode)

        image = 'images.new/%s/%05d.jpg' % (seq.name, objframe.f)
        imgt = T(Image.open(os.path.join(self.dataset_path, image)))

        imgt.rotate_crop(dconf['angle'], objpos, winsize)
        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.resize(self.dataconf.crop_resolution)
        imgt.normalize_affinemap()
        imgframe = normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower'], mode=self.preprocess_mode)

        if dconf['geoocclusion'] is not None:
            geo = dconf['geoocclusion']
            imgframe[geo[0]:geo[2], geo[1]:geo[3], :] = 0.

        """Project pose to the full cropped region."""
        tpose = np.empty(pose_uvd.shape)
        tpose[:, 0:2] = transform_2d_points(imgt.afmat, pose_uvd[:, 0:2],
                transpose=True)
        tpose[:, 2] = (pose_uvd[:, 2] - zrange[0]) / (zrange[1] - zrange[0])

        if imgt.hflip:
            tpose = tpose[pl.map_hflip, :]

        """Set invalid values (-1e9)."""
        if self.remove_outer_joints:
            tpose[tpose < 0] = -1e9
            tpose[tpose > 1] = -1e9

        v = np.expand_dims(get_visible_joints(tpose[:, 0:2]), axis=-1)
        tpose = np.concatenate([tpose, v], axis=-1)

        """Take the last transformation matrix, it should be the same for
        all frames.
        """
        afmat = imgt.afmat.copy()
        output['afmat'] = afmat
        output['aref'] = compute_window_reference(afmat, imgsize)

        """Convert the absolute Z to disparity"""
        rootz = pose_uvd[0:1, 2] / MAX_Z
        # rootz = np.mean(self.zbound) / \
                # np.clip(pose_uvd[0:1, 2], self.zbound[0], self.zbound[1])

        output['camera'] = cam.serialize()
        output['action'] = int(seq.name[1:3]) - 1
        output['pose_w'] = pose_w
        output['pose_uvd'] = pose_uvd
        output['rootz'] = rootz
        output['hflip'] = np.array([imgt.hflip])
        output['pose'] = tpose
        output['frame'] = imgframe
        output['imgsize'] = np.array(imgsize)

        return output


    def load_sequence_pose_annot(self, frames):
        p = np.nan * np.ones((len(frames), h36m23j3d.num_joints, h36m23j3d.dim))

        for i in range(len(frames)):
            p[i, :] = frames[i].pose3d.T[h36m23j3d.map_from_h36m,
                    0:h36m23j3d.dim].copy()

        return p

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.dataconf.input_shape
        if dictkey == 'pose':
            return (self.poselayout.num_joints, self.poselayout.dim+1)
        if dictkey == 'pose_w':
            return (self.poselayout.num_joints, self.poselayout.dim)
        if dictkey == 'pose_uvd':
            return (self.poselayout.num_joints, self.poselayout.dim)
        if dictkey == 'rootz':
            return (1, )
        if dictkey == 'hflip':
            return (1, )
        if dictkey == 'aref':
            return (4,)
        if dictkey == 'action':
            return (1,)
        if dictkey == 'camera':
            return (21,)
        if dictkey == 'afmat':
            return (3, 3)
        if dictkey == 'imgsize':
            return (2,)
        raise Exception('Invalid dictkey `{}` on get_shape!'.format(dictkey))

    def get_length(self, mode):
        return len(self.frame_idx[mode])


class Human36MTest(GenericDataset):
    """Implementation of the Human3.6M dataset for 3D pose estimation, testing
    samples.
    """
    def __init__(self, dataset_path, dataconf,
            bbox_crop_mode='3d',
            preprocess_mode='tf',
            bbox_file_test=None,
            pose_pred_test=None):

        GenericDataset.__init__(self,
                dataset_path=dataset_path,
                dataconf=dataconf,
                poselayout=None,
                preprocess_mode=preprocess_mode)

        self.bbox_pred = None
        self.pose_pred = None

        if bbox_file_test is not None:
            self.bbox_pred = parse_json_bbox(bbox_file_test)

        self.bbox_crop_mode = bbox_crop_mode
        if pose_pred_test is not None:
            self.pose_pred = np.load(pose_pred_test)

        if self.bbox_crop_mode is not None:
            assert hasattr(self, 'pose_pred'), 'If using `bbox_crop_mode` ' \
                    + 'a valid `pose_pred_test` is required!'

        self._load_annotations(os.path.join(dataset_path, 'test_samples2.mat'))

    def _load_annotations(self, filename):
        try:
            self.sequences, self.action_labels, self.joint_labels = \
                    load_h36m_mat_calib_test(filename)

            frame_idx = []
            for s, seq in enumerate(self.sequences):
                for f in range(seq.num_frames):
                    frame_idx.append((s, f))

            self.frame_idx = frame_idx

            global ACTION_LABELS
            ACTION_LABELS = self.action_labels

            self.zbound = ZBOUND

        except:
            warning('Error loading Human3.6M dataset!')
            raise

    def get_meta(self, key, mode=None):
        seq_idx, frame_idx = self.frame_idx[key]
        seq = self.sequences[seq_idx]
        a = int(seq.idname[1:3])
        s = int(seq.idname[5:7])
        e = int(seq.idname[9:11])
        c = int(seq.idname[13:15])
        f = frame_idx + 1

        return (a, s, e, c, f), seq

    def get_data(self, key, mode=None):
        output = {}

        dconf = self.dataconf.get_fixed_config()
        seq_idx, frame_idx = self.frame_idx[key]
        seq = self.sequences[seq_idx]

        filename = os.path.join(self.dataset_path,
                'images.test', seq.sub, seq.name, '%05d.jpg' % (frame_idx + 1))
        imgt = T(Image.open(filename))
        imgsize = imgt.size

        cpar = seq.camera_parameters
        cam = Camera(cpar.R, cpar.T, cpar.f, cpar.c, cpar.p, cpar.k)

        bbox_pred = pose_pred = None
        if self.bbox_pred is not None:
            bbox_pred = self.bbox_pred[key]
        if self.pose_pred is not None:
            pose_pred = self.pose_pred[key]
            pose_pred = cam.project(self.pose_pred[key])

        objpos, winsize, zrange = auto_bbox_cropping(
                gt_pose_uvd=None,
                focal=cam.f,
                box_size_mm=BBOX_REF,
                dconf=dconf,
                imgsize=imgsize,
                bbox_crop_mode=self.bbox_crop_mode,
                bbox_pred=bbox_pred,
                pose_pred=pose_pred,
                mode=TEST_MODE)

        imgt.rotate_crop(dconf['angle'], objpos, winsize)
        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.resize(self.dataconf.crop_resolution)
        imgt.normalize_affinemap()

        afmat = imgt.afmat
        output['frame'] = normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower'], mode=self.preprocess_mode)
        output['afmat'] = afmat.copy()
        output['aref'] = compute_anchors_reference(self.anchors, afmat, imgsize)
        output['camera'] = cam.serialize()
        output['imgsize'] = np.array(imgsize)

        return output

    def get_csv_filepath(self, key):
        seq_idx, frame_idx = self.frame_idx[key]
        seq = self.sequences[seq_idx]
        dpath = os.path.join(seq.sub, seq.name)
        fname = '%05d.csv' % (frame_idx + 1)

        return dpath, fname

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.dataconf.input_shape
        if dictkey == 'aref':
            return (self.num_anchors, 4)
        if dictkey == 'camera':
            return (21,)
        if dictkey == 'afmat':
            return (3, 3)
        if dictkey == 'imgsize':
            return (2,)
        raise Exception('Invalid dictkey `{}` on get_shape!'.format(dictkey))

    def get_length(self, mode=None):
        return len(self.frame_idx)

